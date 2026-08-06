import os
import time
import webbrowser
import pyautogui
import pygetwindow as gw
import keyboard
import pyperclip
import subprocess
import ctypes
from datetime import datetime
import re

def open_chrome_new_tab(url="https://www.google.com"):
    try:
        webbrowser.open_new_tab(url)
        return f"Opened new tab in browser."
    except Exception as e:
        return f"Failed to open browser: {e}"

def take_screenshot():
    try:
        # Create screenshots directory if it doesn't exist
        os.makedirs("screenshots", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/screenshot_{timestamp}.png"
        
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        return f"Screenshot saved as {filename}."
    except Exception as e:
        return f"Failed to take screenshot: {e}"

def open_application(app_name):
    # Mapping common app names to executable commands
    app_commands = {
        "calculator": "calc.exe",
        "notepad": "notepad.exe",
        "paint": "mspaint.exe",
        "explorer": "explorer.exe",
        "cmd": "cmd.exe",
        "word": "winword.exe",
        "excel": "excel.exe"
    }
    
    cmd = app_commands.get(app_name.lower(), app_name)
    try:
        # Popen avoids blocking
        subprocess.Popen(cmd, shell=True)
        return f"Opened {app_name}."
    except Exception as e:
        return f"Failed to open {app_name}: {e}"

def close_active_window():
    try:
        active_window = gw.getActiveWindow()
        if active_window:
            title = active_window.title
            active_window.close()
            return f"Closed {title}."
        else:
            # Fallback using alt+f4
            pyautogui.hotkey('alt', 'f4')
            return "Sent close window command."
    except Exception as e:
        return f"Failed to close window: {e}"

def switch_tab(direction="next"):
    try:
        if direction.lower() in ["next", "right"]:
            pyautogui.hotkey('ctrl', 'tab')
            return "Switched to the next tab."
        elif direction.lower() in ["previous", "left"]:
            pyautogui.hotkey('ctrl', 'shift', 'tab')
            return "Switched to the previous tab."
        else:
            return "Unknown direction for tab switch."
    except Exception as e:
        return f"Failed to switch tab: {e}"

def lock_screen():
    try:
        # Windows API call to lock workstation
        ctypes.windll.user32.LockWorkStation()
        return "Screen locked."
    except Exception as e:
        return f"Failed to lock screen: {e}"

def mute_volume():
    try:
        # Send volume mute media key
        pyautogui.press("volumemute")
        return "Muted the volume."
    except Exception as e:
        return f"Failed to mute volume: {e}"

def set_volume(level="up"):
    try:
        if level.lower() == "up":
            # Press multiple times to make a noticeable difference
            for _ in range(5):
                pyautogui.press("volumeup")
            return "Volume increased."
        elif level.lower() == "down":
            for _ in range(5):
                pyautogui.press("volumedown")
            return "Volume decreased."
        else:
            return "Unknown volume command."
    except Exception as e:
        return f"Failed to change volume: {e}"

def open_file_explorer():
    try:
        subprocess.Popen('explorer.exe')
        return "Opened File Explorer."
    except Exception as e:
        return f"Failed to open File Explorer: {e}"

# ─── COMMAND MATCHING LAYER ──────────────────────────────────

# List of (regex_pattern, handler_function)
AUTOMATION_PATTERNS = [
    (r"take (?:a )?screenshot", lambda m: take_screenshot()),
    (r"open (?:a )?new tab", lambda m: open_chrome_new_tab()),
    (r"open (?:the )?calculator", lambda m: open_application("calculator")),
    (r"open (?:the )?notepad", lambda m: open_application("notepad")),
    (r"open (?:the )?(?:file )?explorer", lambda m: open_file_explorer()),
    (r"close (?:this|active|the current) window", lambda m: close_active_window()),
    (r"switch (?:to )?(?:the )?(next|previous|right|left) tab", lambda m: switch_tab(m.group(1))),
    (r"lock (?:the )?(?:screen|computer|workstation)", lambda m: lock_screen()),
    (r"mute (?:the )?volume|mute", lambda m: mute_volume()),
    (r"(?:turn|volume) (up|down)", lambda m: set_volume(m.group(1)))
]

def match_automation_command(user_input):
    """
    Checks if the user_input matches any automation command.
    Returns the spoken-friendly string result if matched, else None.
    """
    user_input = user_input.lower().strip()
    
    for pattern, handler in AUTOMATION_PATTERNS:
        match = re.search(pattern, user_input)
        if match:
            try:
                return handler(match)
            except Exception as e:
                return f"Error executing command: {e}"
    
    return None

if __name__ == "__main__":
    # Simple test REPL
    print("Automation Matcher Test (type 'exit' to quit)")
    while True:
        cmd = input("> ")
        if cmd.lower() == 'exit':
            break
        result = match_automation_command(cmd)
        if result:
            print(f"Matched! Result: {result}")
        else:
            print("No match.")
