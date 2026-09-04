"""
ARIA v2.0 — AI Orchestrator
────────────────────────────
Plan → Execute → Verify → Synthesize loop.
Turns single model calls into multi-step, multi-model workflows.
Each step is independently routed to the best model.
"""

import re
import time


class Orchestrator:
    """Multi-step task orchestrator with plan-execute-verify loop."""

    MAX_STEPS = 10
    MAX_RETRIES = 1

    def __init__(self, gateway, router, sandbox):
        self.gateway = gateway
        self.router = router
        self.sandbox = sandbox

    def run(self, user_request, system_prompt=None):
        """
        Execute a multi-step orchestrated workflow.

        1. PLAN:      Ask a model to break the request into ordered steps
        2. EXECUTE:   Route each step to the best model and get output
        3. VERIFY:    If a step produces code, run it in the sandbox
        4. SYNTHESIZE: Combine all step results into a final answer

        Args:
            user_request:  The user's original message/task.
            system_prompt: Optional system context.

        Returns:
            OrchestratorResult with plan, steps, synthesis, and metadata.
        """
        t0 = time.time()
        result = OrchestratorResult(original_request=user_request)

        # ── Step 1: PLAN ──────────────────────────────────────
        plan_decision = self.router.route("plan " + user_request)
        plan_prompt = (
            "Break this task into ordered steps. "
            "Output ONLY a numbered list (1. 2. 3. etc). "
            "Each step should be a single, actionable instruction. "
            "Maximum 7 steps. Be concise.\n\n"
            f"Task: {user_request}"
        )

        try:
            plan_response = self.gateway.call(
                plan_decision.model,
                [{"role": "user", "content": plan_prompt}],
            )
            result.plan_raw = plan_response.content
            result.plan_model = plan_decision.model
            steps = self._parse_steps(plan_response.content)
        except Exception as e:
            result.error = f"Planning failed: {e}"
            result.total_time = time.time() - t0
            return result

        if not steps:
            # If planning didn't produce steps, treat as single-step
            steps = [user_request]

        # Cap steps
        steps = steps[:self.MAX_STEPS]
        result.planned_steps = steps

        # ── Step 2 + 3: EXECUTE & VERIFY ──────────────────────
        for i, step_text in enumerate(steps):
            step_result = StepResult(
                index=i + 1,
                instruction=step_text,
            )

            # Route this specific step
            decision = self.router.route(step_text)
            step_result.model = decision.model
            step_result.task_type = decision.task_type
            step_result.routing_reason = decision.reason

            # Build context: include prior step outputs for continuity
            context_messages = self._build_step_context(
                user_request, steps, result.step_results, step_text,
                system_prompt,
            )

            try:
                response = self.gateway.call(decision.model, context_messages)
                step_result.output = response.content
                step_result.latency = response.latency

                # VERIFY: if output contains code, try running it
                if self.sandbox.looks_like_code(response.content):
                    code = self.sandbox.extract_code(response.content)
                    if code:
                        sandbox_result = self.sandbox.run(code)
                        step_result.code_output = sandbox_result

                        # RETRY on failure: re-prompt with the error
                        if not sandbox_result.success and self.MAX_RETRIES > 0:
                            retry_response = self._retry_with_error(
                                decision.model, context_messages,
                                response.content, sandbox_result,
                            )
                            if retry_response:
                                step_result.output = retry_response.content
                                step_result.retried = True
                                # Re-verify
                                retry_code = self.sandbox.extract_code(
                                    retry_response.content
                                )
                                if retry_code:
                                    step_result.code_output = self.sandbox.run(
                                        retry_code
                                    )

            except Exception as e:
                step_result.error = str(e)

            result.step_results.append(step_result)

        # ── Step 4: SYNTHESIZE ────────────────────────────────
        try:
            synthesis = self._synthesize(user_request, result.step_results)
            result.synthesis = synthesis
        except Exception as e:
            # If synthesis fails, concatenate step outputs as fallback
            result.synthesis = "\n\n".join(
                f"**Step {s.index}:** {s.output}"
                for s in result.step_results
                if s.output
            )

        result.total_time = time.time() - t0
        return result

    def _parse_steps(self, plan_text):
        """Extract numbered steps from plan text."""
        # Match lines like "1. Do something" or "1) Do something"
        steps = re.findall(
            r'^\s*\d+[\.\)]\s*(.+?)$',
            plan_text,
            re.MULTILINE,
        )
        # Filter out empty or very short steps
        return [s.strip() for s in steps if len(s.strip()) > 5]

    def _build_step_context(self, original_request, all_steps,
                            completed_steps, current_step, system_prompt):
        """Build the message list for a single step, including prior context."""
        messages = []

        # System context
        sys_content = (
            system_prompt or
            "You are ARIA, an AI assistant executing a multi-step plan."
        )
        sys_content += (
            f"\n\nOriginal user request: {original_request}\n"
            f"You are now working on step: {current_step}\n"
        )

        # Add context from completed steps (last 3 to stay within context)
        if completed_steps:
            prior = completed_steps[-3:]
            sys_content += "\nPrevious step results:\n"
            for s in prior:
                sys_content += f"- Step {s.index}: {s.output[:300]}\n"

        messages.append({"role": "system", "content": sys_content})
        messages.append({"role": "user", "content": current_step})

        return messages

    def _retry_with_error(self, model, original_messages, original_output,
                          sandbox_result):
        """Re-prompt the model with the sandbox error for self-correction."""
        retry_messages = original_messages + [
            {"role": "assistant", "content": original_output},
            {
                "role": "user",
                "content": (
                    f"The code produced an error:\n"
                    f"```\n{sandbox_result.error or sandbox_result.output}\n```\n"
                    f"Please fix the code and try again. "
                    f"Output the corrected code in a ```python block."
                ),
            },
        ]

        try:
            return self.gateway.call(model, retry_messages)
        except Exception:
            return None

    def _synthesize(self, original_request, step_results):
        """Combine all step outputs into a coherent final answer."""
        # If only 1 step, just return its output directly
        if len(step_results) == 1 and step_results[0].output:
            return step_results[0].output

        # Build synthesis prompt with all step results
        steps_summary = ""
        for s in step_results:
            output = s.output or s.error or "(no output)"
            # Include sandbox results if present
            if s.code_output and s.code_output.success:
                output += f"\n\nCode execution result:\n{s.code_output.output}"
            steps_summary += f"\n### Step {s.index}: {s.instruction}\n{output}\n"

        synth_prompt = (
            f"You completed a multi-step task. The original request was:\n"
            f"\"{original_request}\"\n\n"
            f"Here are the results from each step:\n{steps_summary}\n\n"
            f"Now write a clear, cohesive final response that answers the "
            f"original request. Combine and summarize the step results. "
            f"Do NOT mention 'steps' or 'plan' — write as if answering directly."
        )

        # Use the default model for synthesis
        decision = self.router.route("synthesize " + original_request)
        response = self.gateway.call(
            decision.model,
            [{"role": "user", "content": synth_prompt}],
        )
        return response.content


class StepResult:
    """Result of a single orchestrated step."""

    def __init__(self, index, instruction):
        self.index = index
        self.instruction = instruction
        self.model = None
        self.task_type = None
        self.routing_reason = None
        self.output = None
        self.error = None
        self.latency = 0.0
        self.code_output = None   # SandboxResult if code was executed
        self.retried = False

    def to_dict(self):
        d = {
            "step": self.index,
            "instruction": self.instruction,
            "model": self.model,
            "task_type": self.task_type,
            "routing_reason": self.routing_reason,
            "output": self.output,
            "error": self.error,
            "latency": round(self.latency, 2),
            "retried": self.retried,
        }
        if self.code_output:
            d["code_execution"] = self.code_output.to_dict()
        return d


class OrchestratorResult:
    """Full result of an orchestrated workflow."""

    def __init__(self, original_request):
        self.original_request = original_request
        self.plan_raw = None
        self.plan_model = None
        self.planned_steps = []
        self.step_results = []
        self.synthesis = None
        self.error = None
        self.total_time = 0.0

    def to_dict(self):
        return {
            "original_request": self.original_request,
            "plan": {
                "raw": self.plan_raw,
                "model": self.plan_model,
                "steps": self.planned_steps,
            },
            "steps": [s.to_dict() for s in self.step_results],
            "synthesis": self.synthesis,
            "error": self.error,
            "total_time": round(self.total_time, 2),
        }
