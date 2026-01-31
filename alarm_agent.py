import time
import threading
from datetime import datetime
import re

def alarm_thread(time_str, message):
    try:
        alarm_time = datetime.strptime(time_str.strip(), "%Y-%m-%d %H:%M:%S")
        time_diff = (alarm_time - datetime.now()).total_seconds()
        if time_diff > 0:
            time.sleep(time_diff)
        print(f"\n⏰ Alarm! Message: {message}")
    except Exception as e:
        print(f"Alarm thread error: {e}")

def set_alarm(input_str: str) -> str:
    try:
        # Try to extract time and message from a natural language string
        match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', input_str)
        if not match:
            return "Could not find a valid datetime in input."

        time_str = match.group(1)
        message_start = input_str.find(time_str) + len(time_str)
        message = input_str[message_start:].replace("with message", "").strip()
        if not message:
            message = "Alarm!"

        thread = threading.Thread(target=alarm_thread, args=(time_str, message))
        thread.start()

        return f"Alarm set for {time_str} with message: {message}"
    except Exception as e:
        return f"Failed to set alarm: {e}"

def alarm_tool(query: str) -> str:
    # Very basic parser – extend with NLP if needed
    try:
        # Example input: "Set alarm for 2025-05-11 14:00:00 with message Wake Up"
        if "set alarm for" in query.lower():
            parts = query.lower().split("set alarm for")[1].strip().split("with message")
            time_str = parts[0].strip()
            message = parts[1].strip() if len(parts) > 1 else "Alarm!"
            return set_alarm(time_str, message)
        else:
            return "Invalid alarm query format."
    except Exception as e:
        return f"Error setting alarm: {str(e)}"
