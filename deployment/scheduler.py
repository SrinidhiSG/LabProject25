import time
import threading
import subprocess
import sys

def schedule_retraining(interval_days: int = 7):
    """
    Starts a background thread that triggers retraining every `interval_days`.

    Args:
        interval_days (int): Number of days between retraining runs.
    """
    def retrain_loop():
        while True:
            print(f"[Scheduler] Waiting {interval_days} days until next retrain...")
            time.sleep(interval_days * 24 * 3600)  # convert days → seconds

            print("[Scheduler] Triggering automated retraining...")
            subprocess.Popen([sys.executable, "training/training.py"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

    thread = threading.Thread(target=retrain_loop, daemon=True)
    thread.start()
    print("[Scheduler] Automated retraining scheduler started (every "
          f"{interval_days} days).")
