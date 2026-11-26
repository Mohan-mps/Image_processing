"""Resume processing from the last failed image."""

import shutil
from pathlib import Path

# Define paths
input_dir = Path("input")
output_dir = Path("output")
resume_input_dir = Path("input_resume")

# Get all input images
all_images = sorted(input_dir.glob("*.jpg"))

# Get successfully processed images
processed_stems = {p.stem for p in output_dir.glob("*.json")}

# Find images that need processing
to_process = [img for img in all_images if img.stem not in processed_stems]

print(f"Total images in input: {len(all_images)}")
print(f"Already processed: {len(processed_stems)}")
print(f"Remaining to process: {len(to_process)}")
print()

if to_process:
    # Create resume directory
    resume_input_dir.mkdir(exist_ok=True)
    
    print("Images that need processing:")
    for i, img in enumerate(to_process, 1):
        print(f"  {i}. {img.name}")
        # Copy to resume directory
        shutil.copy2(img, resume_input_dir / img.name)
    
    print()
    print(f"✓ Copied {len(to_process)} images to '{resume_input_dir}' directory")
    print()
    print("To resume processing with the FIXED code, run:")
    print(f"  python -m src.main --input {resume_input_dir}")
else:
    print("✓ All images have been processed!")
    if resume_input_dir.exists():
        shutil.rmtree(resume_input_dir)
        print(f"✓ Cleaned up '{resume_input_dir}' directory")
