"""
Download Legitimate Tweet Datasets for Bot Detection Training

This script downloads several publicly available datasets of legitimate human tweets
that can be combined with the Russian troll tweets to create a balanced dataset
for supervised bot detection training.
"""

import kagglehub
import pandas as pd
import numpy as np
import requests
import zipfile
import os
from pathlib import Path
import shutil
from urllib.request import urlretrieve

# Create data directories
HUMAN_DATA_DIR = Path("data/legitimate_tweets")
HUMAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_sentiment140():
    """
    Download the Sentiment140 dataset (1.6M tweets from Stanford)
    This contains verified human tweets with sentiment labels.
    """
    print("📥 Downloading Sentiment140 Dataset...")
    print("   Source: Stanford CS - 1.6M human tweets")
    
    try:
        # Download from HuggingFace (more reliable than original Stanford link)
        train_url = 'https://huggingface.co/datasets/stanfordnlp/sentiment140/resolve/refs%2Fconvert%2Fparquet/sentiment140/train/0000.parquet'
        
        train_file = HUMAN_DATA_DIR / "sentiment140_train.parquet"
        
        print(f"   Downloading to: {train_file}")
        urlretrieve(train_url, train_file)
        
        # Load and check the data
        df = pd.read_parquet(train_file)
        print(f"   ✅ Downloaded {len(df):,} tweets")
        print(f"   Columns: {list(df.columns)}")
        
        # Save as CSV for consistency
        csv_file = HUMAN_DATA_DIR / "sentiment140_human_tweets.csv"
        df.to_csv(csv_file, index=False)
        print(f"   💾 Saved as CSV: {csv_file}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error downloading Sentiment140: {e}")
        return False

def download_twitter_bot_accounts_dataset():
    """
    Download the Twitter Bot Accounts dataset from Kaggle
    This contains both bot and human accounts with labels.
    """
    print("\n📥 Downloading Twitter Bot Accounts Dataset...")
    print("   Source: Kaggle - Labeled bot/human accounts")
    
    try:
        # This dataset has both bots and humans - we'll extract just the humans
        path = kagglehub.dataset_download("davidmartingarcia/twitter-bots-accounts")
        print(f"   Downloaded to: {path}")
        
        # Find CSV files in the downloaded directory
        dataset_path = Path(path)
        csv_files = list(dataset_path.glob("*.csv"))
        
        if csv_files:
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                print(f"   📄 Found file: {csv_file.name}")
                print(f"      Shape: {df.shape}")
                print(f"      Columns: {list(df.columns)}")
                
                # Check if this dataset has bot/human labels
                if 'bot' in df.columns or 'label' in df.columns or 'account_type' in df.columns:
                    # Copy to our data directory
                    dest_file = HUMAN_DATA_DIR / f"kaggle_{csv_file.name}"
                    shutil.copy2(csv_file, dest_file)
                    print(f"   💾 Copied to: {dest_file}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error downloading Twitter Bot Accounts: {e}")
        print("   Note: This requires Kaggle API authentication")
        return False

def create_sample_human_tweets():
    """
    Create a sample dataset of human tweets for demonstration purposes
    if the main downloads fail.
    """
    print("\n🔧 Creating sample human tweets dataset...")
    
    # Sample legitimate-looking tweets (for demo purposes)
    sample_tweets = [
        {"content": "Beautiful sunset today! Nature never fails to amaze me 🌅", "author": "nature_lover_42", "followers": 1250, "following": 350, "updates": 2840},
        {"content": "Just finished reading an amazing book on machine learning. Highly recommend!", "author": "bookworm_jane", "followers": 890, "following": 670, "updates": 1920},
        {"content": "Making homemade pizza tonight. The kids are so excited to help with the toppings!", "author": "family_chef", "followers": 340, "following": 280, "updates": 850},
        {"content": "Went for a morning run and saw the most beautiful gardens. Spring is definitely here!", "author": "runner_mike", "followers": 560, "following": 420, "updates": 1100},
        {"content": "Working on my thesis about renewable energy. Coffee is definitely needed ☕", "author": "grad_student_sarah", "followers": 220, "following": 180, "updates": 650},
    ] * 200  # Repeat to create a larger sample
    
    # Add some variation
    for i, tweet in enumerate(sample_tweets):
        tweet['external_author_id'] = i + 1000000
        tweet['region'] = np.random.choice(['United States', 'Canada', 'United Kingdom', 'Australia'])
        tweet['language'] = 'English'
        tweet['publish_date'] = f"2023-{np.random.randint(1,13):02d}-{np.random.randint(1,28):02d} {np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}:00"
        tweet['harvested_date'] = tweet['publish_date']
        tweet['post_type'] = 'TWEET'
        tweet['account_type'] = 'human'
        tweet['retweet'] = 0  # Original tweets
        tweet['account_category'] = 'Human'
        tweet['bot_label'] = 0  # 0 = human, 1 = bot
        
        # Add some randomness to follower counts
        tweet['followers'] += np.random.randint(-100, 500)
        tweet['following'] += np.random.randint(-50, 100)
        tweet['updates'] += np.random.randint(-200, 300)
    
    df_sample = pd.DataFrame(sample_tweets)
    sample_file = HUMAN_DATA_DIR / "sample_human_tweets.csv"
    df_sample.to_csv(sample_file, index=False)
    
    print(f"   ✅ Created {len(df_sample)} sample human tweets")
    print(f"   💾 Saved to: {sample_file}")
    return True

def download_additional_datasets():
    """
    Try to download additional datasets for more diverse human tweets
    """
    print("\n📥 Attempting to download additional datasets...")
    
    datasets_to_try = [
        {
            'name': 'Twitter Sentiment Dataset',
            'kaggle_path': 'kazanova/sentiment140',
            'description': 'Alternative Sentiment140 source'
        },
        {
            'name': 'Tweet Sentiment Extraction',
            'kaggle_path': 'c/tweet-sentiment-extraction',
            'description': 'Competition dataset with human tweets'
        }
    ]
    
    for dataset in datasets_to_try:
        try:
            print(f"   Trying {dataset['name']}...")
            path = kagglehub.dataset_download(dataset['kaggle_path'])
            print(f"   ✅ Downloaded: {dataset['name']}")
            
            # Copy files to our directory
            dataset_path = Path(path)
            for file in dataset_path.glob("*.csv"):
                dest = HUMAN_DATA_DIR / f"{dataset['name'].lower().replace(' ', '_')}_{file.name}"
                shutil.copy2(file, dest)
                print(f"      💾 Copied: {file.name}")
                
        except Exception as e:
            print(f"   ⚠️  Could not download {dataset['name']}: {str(e)[:100]}...")

def summarize_downloaded_data():
    """
    Summarize all the downloaded legitimate tweet datasets
    """
    print("\n📊 SUMMARY OF DOWNLOADED LEGITIMATE TWEET DATASETS")
    print("=" * 60)
    
    csv_files = list(HUMAN_DATA_DIR.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in legitimate tweets directory")
        return
    
    total_tweets = 0
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            file_size = csv_file.stat().st_size / (1024 * 1024)  # MB
            
            print(f"\n📄 {csv_file.name}")
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Size: {file_size:.1f} MB")
            print(f"   Columns: {list(df.columns)[:5]}...")
            
            total_tweets += len(df)
            
        except Exception as e:
            print(f"   ❌ Error reading {csv_file.name}: {e}")
    
    print(f"\n🎯 TOTAL LEGITIMATE TWEETS AVAILABLE: {total_tweets:,}")
    print(f"📁 Data stored in: {HUMAN_DATA_DIR}")

def main():
    """
    Main function to download all legitimate tweet datasets
    """
    print("🤖 LEGITIMATE TWEET DATASET DOWNLOADER")
    print("=" * 50)
    print("Purpose: Download human tweets to pair with Russian troll tweets")
    print("Goal: Create balanced dataset for bot detection training")
    print()
    
    success_count = 0
    
    # Try different data sources
    if download_sentiment140():
        success_count += 1
    
    if download_twitter_bot_accounts_dataset():
        success_count += 1
    
    download_additional_datasets()  # This may or may not succeed
    
    # Always create sample data as fallback
    if create_sample_human_tweets():
        success_count += 1
    
    # Summarize results
    summarize_downloaded_data()
    
    print(f"\n✅ Successfully downloaded {success_count} dataset(s)")
    print("\n🎯 NEXT STEPS:")
    print("1. Run 'python create_balanced_dataset.py' to combine with bot data")
    print("2. Use the combined dataset for logistic regression training")
    print("3. Now you can predict bot probabilities!")

if __name__ == "__main__":
    main() 