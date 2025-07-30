import kagglehub
import shutil
import os
from pathlib import Path

# Download latest version
path = kagglehub.dataset_download("fivethirtyeight/russian-troll-tweets")

print("Path to dataset files:", path)

# Copy files to current directory
current_dir = Path.cwd()
dataset_dir = Path(path)

print(f"Copying files to: {current_dir}")

# Copy all CSV files to current directory
for csv_file in dataset_dir.glob("*.csv"):
    destination = current_dir / csv_file.name
    print(f"Copying {csv_file.name}...")
    shutil.copy2(csv_file, destination)
    
print("Dataset files copied to current directory!")

# List the copied files
print("\nFiles in current directory:")
for csv_file in current_dir.glob("IRAhandle_tweets_*.csv"):
    file_size = csv_file.stat().st_size / (1024*1024)  # Size in MB
    print(f"  {csv_file.name} ({file_size:.1f} MB)") 