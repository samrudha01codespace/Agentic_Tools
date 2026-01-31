import subprocess
import psutil
import time
import json
from datetime import datetime

LOG_FILE = "open_apps_log.json"
CHECK_INTERVAL = 10  # seconds

tracked_windows = {}


def get_all_visible_windows():
    try:
        output = subprocess.check_output(["wmctrl", "-lp"]).decode().splitlines()
        windows = []

        for line in output:
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue

            window_id, _, pid, _, title = parts
            pid = int(pid)

            # Filter out minimized or invisible windows
            xprop_output = subprocess.check_output(["xprop", "-id", window_id])
            if b"_NET_WM_STATE_HIDDEN" in xprop_output:
                continue  # window is hidden/minimized

            try:
                proc = psutil.Process(pid)
                windows.append({
                    "pid": pid,
                    "name": proc.name(),
                    "cmd": " ".join(proc.cmdline()),
                    "window_title": title
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return windows
    except Exception as e:
        print(f"[Error] Failed to get windows: {e}")
        return []



def save_log(log_data):
    with open(LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=2)


def track_open_apps():
    global tracked_windows
    print("[Jarvis] Tracking all open GUI apps on Ubuntu, Sir.")

    app_log = []
    try:
        while True:
            current_windows = get_all_visible_windows()
            now = datetime.now()

            current_pids = set()

            for win in current_windows:
                pid = win["pid"]
                current_pids.add(pid)

                if pid not in tracked_windows:
                    tracked_windows[pid] = {
                        "name": win["name"],
                        "window_title": win["window_title"],
                        "cmd": win["cmd"],
                        "start_time": now
                    }

            # Check for closed windows
            closed_pids = [pid for pid in tracked_windows if pid not in current_pids]
            for pid in closed_pids:
                info = tracked_windows[pid]
                end = now
                duration = (end - info["start_time"]).total_seconds() / 60
                app_log.append({
                    "app": info["name"],
                    "window_title": info["window_title"],
                    "cmd": info["cmd"],
                    "start_time": info["start_time"].isoformat(),
                    "end_time": end.isoformat(),
                    "duration_minutes": round(duration, 2)
                })
                print(f"[Jarvis] Logged closed app: {info['name']} - {info['window_title']}")
                del tracked_windows[pid]

            save_log(app_log)
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n[Jarvis] App monitoring stopped.")
        now = datetime.now()
        for pid, info in tracked_windows.items():
            duration = (now - info["start_time"]).total_seconds() / 60
            app_log.append({
                "app": info["name"],
                "window_title": info["window_title"],
                "cmd": info["cmd"],
                "start_time": info["start_time"].isoformat(),
                "end_time": now.isoformat(),
                "duration_minutes": round(duration, 2)
            })
        save_log(app_log)


if __name__ == "__main__":
    track_open_apps()
