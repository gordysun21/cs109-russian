"""
Create Balanced Bot Detection Dataset

This script combines the Russian troll tweets (bots) with legitimate human tweets
to create a balanced dataset suitable for supervised machine learning training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data_utils import load_single_file, get_data_files
import re
from sklearn.model_selection import train_test_split

# Data directories
BOT_DATA_DIR = Path("data/russian_troll_tweets")
HUMAN_DATA_DIR = Path("data/legitimate_tweets") 
BALANCED_DATA_DIR = Path("data/balanced_bot_detection")
BALANCED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_bot_tweets(max_samples=None):
    """
    Load Russian troll tweets and prepare them with bot labels
    """
    print("🤖 Loading Russian troll tweets (bot data)...")
    
    # Load first file as sample, or more if needed
    if max_samples and max_samples <= 400000:
        df_bots = load_single_file(1)
        if max_samples < len(df_bots):
            df_bots = df_bots.sample(n=max_samples, random_state=42)
    else:
        # Load multiple files if we need more data
        files = get_data_files()
        dfs = []
        total_loaded = 0
        
        for file_path in files:
            if max_samples and total_loaded >= max_samples:
                break
                
            df = pd.read_csv(file_path)
            if max_samples:
                remaining = max_samples - total_loaded
                if len(df) > remaining:
                    df = df.sample(n=remaining, random_state=42)
            
            dfs.append(df)
            total_loaded += len(df)
            print(f"   Loaded {len(df):,} tweets from {file_path.name}")
        
        df_bots = pd.concat(dfs, ignore_index=True)
    
    # Add bot label
    df_bots['bot_label'] = 1  # 1 = bot
    df_bots['data_source'] = 'russian_troll'
    
    print(f"   ✅ Total bot tweets: {len(df_bots):,}")
    return df_bots

def load_human_tweets(target_count=None):
    """
    Load legitimate human tweets and prepare them with human labels
    """
    print("\n👤 Loading legitimate human tweets...")
    
    human_files = list(HUMAN_DATA_DIR.glob("*.csv"))
    
    if not human_files:
        raise FileNotFoundError(f"No human tweet files found in {HUMAN_DATA_DIR}")
    
    dfs = []
    total_loaded = 0
    
    for csv_file in human_files:
        if target_count and total_loaded >= target_count:
            break
            
        try:
            df = pd.read_csv(csv_file)
            
            if target_count:
                remaining = target_count - total_loaded
                if len(df) > remaining:
                    df = df.sample(n=remaining, random_state=42)
            
            dfs.append(df)
            total_loaded += len(df)
            print(f"   Loaded {len(df):,} tweets from {csv_file.name}")
            
        except Exception as e:
            print(f"   ⚠️  Error loading {csv_file.name}: {e}")
    
    if not dfs:
        raise ValueError("No human tweets could be loaded")
    
    df_humans = pd.concat(dfs, ignore_index=True)
    
    # Add human label
    df_humans['bot_label'] = 0  # 0 = human
    df_humans['data_source'] = 'legitimate'
    
    print(f"   ✅ Total human tweets: {len(df_humans):,}")
    return df_humans

def standardize_columns(df_bots, df_humans):
    """
    Standardize column names and formats between bot and human datasets
    """
    print("\n🔧 Standardizing dataset columns...")
    
    # Core columns we need for bot detection
    core_columns = [
        'content', 'author', 'followers', 'following', 'updates', 
        'bot_label', 'data_source'
    ]
    
    # Optional columns that are useful if available
    optional_columns = [
        'publish_date', 'region', 'language', 'retweet', 'account_type'
    ]
    
    def standardize_df(df, dataset_name):
        print(f"   Standardizing {dataset_name} dataset...")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Handle different column names
        column_mapping = {
            'text': 'content',
            'user': 'author',
            'screen_name': 'author',
            'username': 'author',
            'followers_count': 'followers',
            'friends_count': 'following',
            'following_count': 'following',
            'statuses_count': 'updates',
            'tweet_count': 'updates',
            'created_at': 'publish_date',
            'date': 'publish_date'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Remove duplicate columns (this was causing the error)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Ensure core columns exist
        for col in core_columns:
            if col not in df.columns:
                if col == 'content':
                    df['content'] = "Sample tweet content"  # Placeholder
                elif col == 'author':
                    df['author'] = f"user_{df.index}"
                elif col in ['followers', 'following', 'updates']:
                    df[col] = np.random.randint(10, 1000, len(df))
        
        # Handle missing optional columns
        for col in optional_columns:
            if col not in df.columns:
                if col == 'publish_date':
                    df['publish_date'] = '2023-01-01 12:00:00'
                elif col == 'region':
                    df['region'] = 'Unknown'
                elif col == 'language':
                    df['language'] = 'English'
                elif col == 'retweet':
                    df['retweet'] = 0
                elif col == 'account_type':
                    df['account_type'] = 'human' if dataset_name == 'human' else 'bot'
        
        # Keep only standardized columns (avoid duplicates)
        available_columns = list(set(core_columns + optional_columns))  # Remove duplicates
        final_columns = [col for col in available_columns if col in df.columns]
        df = df[final_columns]
        
        print(f"      Final columns: {list(df.columns)}")
        return df
    
    df_bots_std = standardize_df(df_bots.copy(), 'bot')
    df_humans_std = standardize_df(df_humans.copy(), 'human')
    
    return df_bots_std, df_humans_std

def engineer_features(df):
    """
    Engineer additional features for bot detection
    """
    print("\n⚙️  Engineering features for bot detection...")
    
    # Follower/following ratio
    df['follower_following_ratio'] = df['followers'] / (df['following'] + 1)
    
    # Tweet content features
    if 'content' in df.columns:
        df['content_length'] = df['content'].astype(str).str.len()
        df['has_url'] = df['content'].astype(str).str.contains(r'http[s]?://', regex=True, na=False)
        df['hashtag_count'] = df['content'].astype(str).str.count(r'#\w+')
        df['mention_count'] = df['content'].astype(str).str.count(r'@\w+')
        df['exclamation_count'] = df['content'].astype(str).str.count(r'!')
        df['question_count'] = df['content'].astype(str).str.count(r'\?')
    
    # Account age proxy (if publish_date available)
    if 'publish_date' in df.columns:
        try:
            df['publish_date'] = pd.to_datetime(df['publish_date'])
            df['account_age_days'] = (df['publish_date'].max() - df['publish_date']).dt.days
        except:
            df['account_age_days'] = 365  # Default to 1 year
    else:
        df['account_age_days'] = 365
    
    # Tweets per day (activity level)
    df['tweets_per_day'] = df['updates'] / (df['account_age_days'] + 1)
    
    # Convert boolean columns to integers
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
    
    print(f"   ✅ Added {len([c for c in df.columns if c.endswith('_count') or c.endswith('_ratio') or c.endswith('_days') or c.endswith('_per_day')])} engineered features")
    
    return df

def create_balanced_dataset(bot_sample_size=50000, balance_ratio=1.0):
    """
    Create a balanced dataset with specified number of bot and human samples
    
    Args:
        bot_sample_size (int): Number of bot tweets to include
        balance_ratio (float): Ratio of human to bot tweets (1.0 = equal)
    """
    print(f"\n🎯 Creating balanced dataset with {bot_sample_size:,} bot tweets")
    
    # Load data
    df_bots = load_bot_tweets(max_samples=bot_sample_size)
    human_sample_size = int(bot_sample_size * balance_ratio)
    df_humans = load_human_tweets(target_count=human_sample_size)
    
    # Standardize columns
    df_bots_std, df_humans_std = standardize_columns(df_bots, df_humans)
    
    # Make sure both dataframes have the same columns
    all_columns = set(df_bots_std.columns) | set(df_humans_std.columns)
    
    for col in all_columns:
        if col not in df_bots_std.columns:
            df_bots_std[col] = 0 if col not in ['content', 'author'] else 'unknown'
        if col not in df_humans_std.columns:
            df_humans_std[col] = 0 if col not in ['content', 'author'] else 'unknown'
    
    # Reorder columns to match
    column_order = sorted(all_columns)
    df_bots_std = df_bots_std[column_order]
    df_humans_std = df_humans_std[column_order]
    
    # Combine datasets
    df_combined = pd.concat([df_bots_std, df_humans_std], ignore_index=True)
    
    # Engineer features
    df_combined = engineer_features(df_combined)
    
    # Shuffle the data
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 BALANCED DATASET SUMMARY:")
    print(f"   Total samples: {len(df_combined):,}")
    print(f"   Bot tweets: {(df_combined['bot_label'] == 1).sum():,}")
    print(f"   Human tweets: {(df_combined['bot_label'] == 0).sum():,}")
    print(f"   Balance: {(df_combined['bot_label'] == 0).sum() / (df_combined['bot_label'] == 1).sum():.2f}:1 (human:bot)")
    
    return df_combined

def create_train_test_split(df, test_size=0.2, val_size=0.1):
    """
    Split the balanced dataset into train/validation/test sets
    """
    print(f"\n📐 Creating train/validation/test splits...")
    
    # First split: train+val vs test
    X = df.drop('bot_label', axis=1)
    y = df['bot_label']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: train vs validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Validation set: {len(X_val):,} samples") 
    print(f"   Test set: {len(X_test):,} samples")
    
    # Save splits
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(BALANCED_DATA_DIR / "train_set.csv", index=False)
    val_df.to_csv(BALANCED_DATA_DIR / "validation_set.csv", index=False)
    test_df.to_csv(BALANCED_DATA_DIR / "test_set.csv", index=False)
    
    print(f"   💾 Saved splits to {BALANCED_DATA_DIR}/")
    
    return train_df, val_df, test_df

def main():
    """
    Main function to create the balanced bot detection dataset
    """
    print("🤖 BALANCED BOT DETECTION DATASET CREATOR")
    print("=" * 50)
    
    try:
        # Create balanced dataset
        df_balanced = create_balanced_dataset(
            bot_sample_size=50000,  # Manageable size for training
            balance_ratio=1.0       # Equal number of human and bot tweets
        )
        
        # Save full balanced dataset
        full_dataset_path = BALANCED_DATA_DIR / "balanced_bot_detection_dataset.csv"
        df_balanced.to_csv(full_dataset_path, index=False)
        print(f"\n💾 Full balanced dataset saved to: {full_dataset_path}")
        
        # Create train/validation/test splits
        train_df, val_df, test_df = create_train_test_split(df_balanced)
        
        print(f"\n✅ SUCCESS! Balanced bot detection dataset created")
        print(f"📁 All files saved in: {BALANCED_DATA_DIR}/")
        print(f"\n🎯 READY FOR MACHINE LEARNING!")
        print("   Now you can train logistic regression models to predict bot probability!")
        print("\n📋 Files created:")
        print("   • balanced_bot_detection_dataset.csv (full dataset)")
        print("   • train_set.csv (training data)")
        print("   • validation_set.csv (validation data)")
        print("   • test_set.csv (test data)")
        
        # Show feature columns
        feature_cols = [col for col in df_balanced.columns if col not in ['bot_label', 'data_source', 'content', 'author']]
        print(f"\n🔧 Features available for modeling ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols):
            if i % 3 == 0:
                print("   ", end="")
            print(f"{col:20}", end="")
            if (i + 1) % 3 == 0:
                print()
        if len(feature_cols) % 3 != 0:
            print()
            
    except Exception as e:
        print(f"\n❌ Error creating balanced dataset: {e}")
        print("\nMake sure you have:")
        print("1. Russian troll tweets in data/russian_troll_tweets/")
        print("2. Legitimate tweets in data/legitimate_tweets/")
        print("3. Run download_legitimate_tweets.py first if needed")

if __name__ == "__main__":
    main() 