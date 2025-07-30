"""
Data Utilities for Russian Troll Tweets Dataset

This module provides convenient functions for loading and working with 
the Russian troll tweets dataset stored in data/russian_troll_tweets/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

# Data directory constants
DATA_DIR = Path("data/russian_troll_tweets")
CSV_PATTERN = "IRAhandle_tweets_*.csv"

def get_data_files() -> List[Path]:
    """
    Get list of all CSV files in the dataset directory.
    
    Returns:
        List[Path]: List of paths to CSV files
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    
    files = list(DATA_DIR.glob(CSV_PATTERN))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    
    return sorted(files)

def load_single_file(file_number: int = 1) -> pd.DataFrame:
    """
    Load a single CSV file from the dataset.
    
    Args:
        file_number (int): File number to load (1-9)
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if not 1 <= file_number <= 9:
        raise ValueError("File number must be between 1 and 9")
    
    file_path = DATA_DIR / f"IRAhandle_tweets_{file_number}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path)

def load_all_files(max_files: Optional[int] = None) -> pd.DataFrame:
    """
    Load and concatenate all CSV files in the dataset.
    
    Args:
        max_files (Optional[int]): Maximum number of files to load (for testing)
        
    Returns:
        pd.DataFrame: Combined dataset from all files
    """
    files = get_data_files()
    
    if max_files:
        files = files[:max_files]
    
    print(f"Loading {len(files)} files...")
    
    dataframes = []
    for i, file_path in enumerate(files, 1):
        print(f"  Loading file {i}/{len(files)}: {file_path.name}")
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"✅ Loaded {len(combined_df):,} total tweets from {len(files)} files")
    
    return combined_df

def get_dataset_info() -> dict:
    """
    Get basic information about the dataset files.
    
    Returns:
        dict: Information about dataset files
    """
    files = get_data_files()
    
    info = {
        'num_files': len(files),
        'file_names': [f.name for f in files],
        'total_size_mb': sum(f.stat().st_size for f in files) / (1024 * 1024),
        'data_directory': str(DATA_DIR)
    }
    
    return info

def load_sample(n_samples: int = 1000, file_number: int = 1, random_state: int = 42) -> pd.DataFrame:
    """
    Load a random sample from the dataset for quick analysis.
    
    Args:
        n_samples (int): Number of samples to load
        file_number (int): Which file to sample from (1-9)
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Random sample from the dataset
    """
    df = load_single_file(file_number)
    
    if n_samples >= len(df):
        print(f"Requested {n_samples} samples, but file only has {len(df)} rows. Returning all rows.")
        return df
    
    return df.sample(n=n_samples, random_state=random_state)

def print_dataset_summary():
    """Print a summary of the dataset structure and contents."""
    print("📊 Russian Troll Tweets Dataset Summary")
    print("=" * 40)
    
    try:
        info = get_dataset_info()
        print(f"📁 Data Directory: {info['data_directory']}")
        print(f"📄 Number of files: {info['num_files']}")
        print(f"💾 Total size: {info['total_size_mb']:.1f} MB")
        print(f"📋 Files: {', '.join(info['file_names'])}")
        
        # Load first file for structure info
        df_sample = load_single_file(1)
        print(f"\n📊 Data Structure (from first file):")
        print(f"   Rows: {len(df_sample):,}")
        print(f"   Columns: {len(df_sample.columns)}")
        print(f"   Column names: {list(df_sample.columns)}")
        
        # Date range
        if 'publish_date' in df_sample.columns:
            df_sample['publish_date'] = pd.to_datetime(df_sample['publish_date'])
            print(f"📅 Date range: {df_sample['publish_date'].min()} to {df_sample['publish_date'].max()}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("   Please run copy_dataset.py first to download the data.")

if __name__ == "__main__":
    # Demo usage
    print_dataset_summary() 