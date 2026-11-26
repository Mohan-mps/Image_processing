"""Monitor processing progress in real-time."""

import time
from pathlib import Path
from datetime import datetime

output_dir = Path("output")
input_resume_dir = Path("input_resume")

# Count files
total_to_process = len(list(input_resume_dir.glob("*.jpg"))) if input_resume_dir.exists() else 0
already_done_before = 4  # Files processed in first run

print("=" * 60)
print("ChartXtract Processing Monitor")
print("=" * 60)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Images from first run: {already_done_before}")
print(f"Images to process now: {total_to_process}")
print(f"Total target: {already_done_before + total_to_process}")
print("=" * 60)
print()

last_count = 0
start_time = time.time()

try:
    while True:
        # Count JSON files in output
        current_count = len(list(output_dir.glob("*.json")))
        new_in_this_run = current_count - already_done_before
        
        if new_in_this_run != last_count:
            elapsed = time.time() - start_time
            rate = new_in_this_run / elapsed if elapsed > 0 else 0
            remaining = total_to_process - new_in_this_run
            eta_seconds = remaining / rate if rate > 0 else 0
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Progress: {new_in_this_run}/{total_to_process} "
                  f"({100*new_in_this_run/total_to_process:.1f}%) | "
                  f"Total: {current_count}/{already_done_before + total_to_process} | "
                  f"Rate: {rate*60:.1f} files/hour | "
                  f"ETA: {eta_seconds/60:.1f} min")
            
            last_count = new_in_this_run
            
            if new_in_this_run >= total_to_process:
                print()
                print("=" * 60)
                print("✓ ALL IMAGES PROCESSED!")
                print(f"Total time: {elapsed/60:.1f} minutes")
                print("=" * 60)
                break
        
        time.sleep(10)  # Check every 10 seconds

except KeyboardInterrupt:
    print("\n\nMonitoring stopped by user.")
    print(f"Final count: {new_in_this_run}/{total_to_process} processed")
