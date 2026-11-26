"""Process multiple images in parallel using multiple Python processes."""

import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

def process_batch(batch_id, image_files):
    """Process a batch of images in a separate process."""
    # Create batch-specific directories
    batch_input = Path(f"input_batch_{batch_id}")
    batch_output = Path(f"output_batch_{batch_id}")
    batch_crops = Path(f"crops_batch_{batch_id}")
    batch_axis = Path(f"axis_batch_{batch_id}")
    
    batch_input.mkdir(exist_ok=True)
    batch_output.mkdir(exist_ok=True)
    batch_crops.mkdir(exist_ok=True)
    batch_axis.mkdir(exist_ok=True)
    
    # Copy images to batch input
    for img in image_files:
        shutil.copy2(img, batch_input / img.name)
    
    # Run processing
    cmd = [
        sys.executable, "-m", "src.main",
        "--input", str(batch_input),
        "--output", str(batch_output),
        "--crops", str(batch_crops),
        "--axis-crops", str(batch_axis),
        "--ollama-timeout", "90.0",  # Faster timeout
        "--ollama-retries", "3",     # Fewer retries since it's working
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Copy results back to main output
    output_dir = Path("output")
    for json_file in batch_output.glob("*.json"):
        shutil.copy2(json_file, output_dir / json_file.name)
    
    # Cleanup batch directories
    shutil.rmtree(batch_input, ignore_errors=True)
    shutil.rmtree(batch_output, ignore_errors=True)
    shutil.rmtree(batch_crops, ignore_errors=True)
    shutil.rmtree(batch_axis, ignore_errors=True)
    
    return batch_id, len(image_files), result.returncode


def main():
    input_dir = Path("input_resume")
    output_dir = Path("output")
    
    # Get remaining images
    all_images = sorted(input_dir.glob("*.jpg"))
    processed = {p.stem for p in output_dir.glob("*.json")}
    remaining = [img for img in all_images if img.stem not in processed]
    
    if not remaining:
        print("All images already processed!")
        return
    
    print(f"Images remaining: {len(remaining)}")
    
    # Split into batches (number of parallel processes)
    num_processes = 3  # Process 3 images simultaneously
    batch_size = max(1, len(remaining) // num_processes)
    
    batches = []
    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i + batch_size]
        if batch:
            batches.append(batch)
    
    print(f"Splitting into {len(batches)} batches")
    print(f"Processing {num_processes} images in parallel...")
    print()
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(process_batch, i, batch): i 
            for i, batch in enumerate(batches)
        }
        
        for future in as_completed(futures):
            batch_id = futures[future]
            try:
                batch_id, count, returncode = future.result()
                if returncode == 0:
                    print(f"✓ Batch {batch_id} completed ({count} images)")
                else:
                    print(f"✗ Batch {batch_id} failed with code {returncode}")
            except Exception as e:
                print(f"✗ Batch {batch_id} error: {e}")
    
    print()
    print("All batches completed!")


if __name__ == "__main__":
    main()
