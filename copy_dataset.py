import shutil
import os
from pathlib import Path

# Path where kagglehub cached the dataset
cache_path = Path("/Users/gordysun/.cache/kagglehub/datasets/fivethirtyeight/russian-troll-tweets/versions/2")
# Create data directory structure
data_dir = Path("data/russian_troll_tweets")
data_dir.mkdir(parents=True, exist_ok=True)

print(f"Copying files from: {cache_path}")
print(f"Copying files to: {data_dir}")

# Check if cache directory exists
if not cache_path.exists():
    print("ERROR: Cache directory not found. Please run the download script first.")
    exit(1)

# Copy all CSV files to data directory
copied_files = []
for csv_file in cache_path.glob("*.csv"):
    destination = data_dir / csv_file.name
    print(f"Copying {csv_file.name}...")
    shutil.copy2(csv_file, destination)
    copied_files.append(csv_file.name)
    
print(f"\n✅ Successfully copied {len(copied_files)} files to data directory!")

# List the copied files with sizes
print("\nFiles in data/russian_troll_tweets directory:")
total_size = 0
for csv_file in data_dir.glob("IRAhandle_tweets_*.csv"):
    file_size = csv_file.stat().st_size / (1024*1024)  # Size in MB
    total_size += file_size
    print(f"  {csv_file.name} ({file_size:.1f} MB)")

print(f"\nTotal dataset size: {total_size:.1f} MB") 