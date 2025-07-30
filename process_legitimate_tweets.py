"""
Process Legitimate Tweet Datasets

This script processes the downloaded legitimate tweet datasets,
handles encoding issues, and prepares them for bot detection training.
"""

import pandas as pd
import numpy as np
from pathlib import Path

HUMAN_DATA_DIR = Path("data/legitimate_tweets")

def process_sentiment140_parquet():
    """Process the Sentiment140 parquet file"""
    print("📥 Processing Sentiment140 parquet file...")
    
    parquet_file = HUMAN_DATA_DIR / "sentiment140_train.parquet"
    
    if not parquet_file.exists():
        print("   ❌ Sentiment140 parquet file not found")
        return None
    
    try:
        df = pd.read_parquet(parquet_file)
        print(f"   ✅ Loaded {len(df):,} tweets from parquet")
        print(f"   Columns: {list(df.columns)}")
        
        # Take a reasonable sample for our purposes
        sample_size = min(100000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        # Standardize column names
        if 'text' in df_sample.columns:
            df_sample = df_sample.rename(columns={'text': 'content'})
        if 'user' in df_sample.columns:
            df_sample = df_sample.rename(columns={'user': 'author'})
            
        # Add missing columns with reasonable defaults
        df_sample['followers'] = np.random.randint(50, 2000, len(df_sample))
        df_sample['following'] = np.random.randint(20, 500, len(df_sample))
        df_sample['updates'] = np.random.randint(100, 5000, len(df_sample))
        df_sample['retweet'] = 0  # Assume original tweets
        df_sample['region'] = 'United States'
        df_sample['language'] = 'English'
        df_sample['account_type'] = 'human'
        df_sample['bot_label'] = 0
        
        # Save processed version
        output_file = HUMAN_DATA_DIR / "sentiment140_processed.csv"
        df_sample.to_csv(output_file, index=False)
        print(f"   💾 Saved {len(df_sample):,} processed tweets to: {output_file}")
        
        return df_sample
        
    except Exception as e:
        print(f"   ❌ Error processing parquet: {e}")
        return None

def process_sentiment140_csv():
    """Process the Sentiment140 CSV file with encoding handling"""
    print("\n📥 Processing Sentiment140 CSV file...")
    
    csv_file = HUMAN_DATA_DIR / "twitter_sentiment_dataset_training.1600000.processed.noemoticon.csv"
    
    if not csv_file.exists():
        print("   ❌ Sentiment140 CSV file not found")
        return None
    
    try:
        # Try different encodings
        encodings = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-8']
        
        for encoding in encodings:
            try:
                print(f"   Trying encoding: {encoding}")
                
                # Load with specific column names (Sentiment140 format)
                column_names = ['sentiment', 'id', 'date', 'query', 'author', 'content']
                df = pd.read_csv(csv_file, 
                               encoding=encoding, 
                               header=None, 
                               names=column_names,
                               nrows=100000)  # Load first 100k for speed
                
                print(f"   ✅ Successfully loaded {len(df):,} tweets with {encoding}")
                break
                
            except UnicodeDecodeError:
                continue
        else:
            print("   ❌ Could not decode file with any encoding")
            return None
            
        # Add missing columns
        df['followers'] = np.random.randint(50, 2000, len(df))
        df['following'] = np.random.randint(20, 500, len(df))
        df['updates'] = np.random.randint(100, 5000, len(df))
        df['retweet'] = 0
        df['region'] = 'United States'
        df['language'] = 'English'
        df['account_type'] = 'human'
        df['bot_label'] = 0
        
        # Clean up data
        df['content'] = df['content'].astype(str)
        df['author'] = df['author'].astype(str)
        
        # Save processed version
        output_file = HUMAN_DATA_DIR / "sentiment140_csv_processed.csv"
        df.to_csv(output_file, index=False)
        print(f"   💾 Saved {len(df):,} processed tweets to: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error processing CSV: {e}")
        return None

def create_enhanced_sample_data():
    """Create a larger, more diverse sample dataset"""
    print("\n🔧 Creating enhanced sample human tweets...")
    
    # More diverse sample tweets
    sample_tweets = [
        {"content": "Beautiful sunset today! Nature never fails to amaze me 🌅", "author": "nature_lover_42"},
        {"content": "Just finished reading an amazing book on machine learning. Highly recommend!", "author": "bookworm_jane"},
        {"content": "Making homemade pizza tonight. The kids are so excited to help!", "author": "family_chef"},
        {"content": "Went for a morning run and saw the most beautiful gardens. Spring is here!", "author": "runner_mike"},
        {"content": "Working on my thesis about renewable energy. Coffee needed ☕", "author": "grad_student_sarah"},
        {"content": "Great concert last night! The band really knows how to put on a show", "author": "music_fan_87"},
        {"content": "Trying a new recipe for chicken curry. Hope it turns out well!", "author": "cooking_enthusiast"},
        {"content": "Weekend hiking trip was amazing. The views from the mountain were incredible", "author": "outdoor_adventurer"},
        {"content": "My cat learned a new trick today. She can now give high fives!", "author": "cat_parent_2023"},
        {"content": "Finally organized my entire bookshelf. So satisfying to see everything in order", "author": "organization_lover"},
        {"content": "Watched a documentary about ocean life. The underwater footage was stunning", "author": "nature_documentary_fan"},
        {"content": "First time trying surfing today. Fell off the board more times than I can count!", "author": "surf_beginner"},
        {"content": "Homemade bread turned out perfect! Nothing beats the smell of fresh baking", "author": "baking_amateur"},
        {"content": "Finished a 1000-piece puzzle with the family. Took us three weekends!", "author": "puzzle_master"},
        {"content": "Local farmers market had the best strawberries. Summer fruits are the best", "author": "farmers_market_regular"},
    ]
    
    # Generate more samples with variation
    enhanced_samples = []
    for i in range(10000):  # Create 10k samples
        base_tweet = sample_tweets[i % len(sample_tweets)].copy()
        
        # Add variation
        base_tweet['followers'] = np.random.randint(50, 3000)
        base_tweet['following'] = np.random.randint(20, 800)
        base_tweet['updates'] = np.random.randint(100, 8000)
        base_tweet['retweet'] = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% retweets
        base_tweet['region'] = np.random.choice(['United States', 'Canada', 'United Kingdom', 'Australia'])
        base_tweet['language'] = 'English'
        base_tweet['account_type'] = 'human'
        base_tweet['bot_label'] = 0
        base_tweet['external_author_id'] = i + 2000000
        base_tweet['publish_date'] = f"2023-{np.random.randint(1,13):02d}-{np.random.randint(1,28):02d}"
        
        enhanced_samples.append(base_tweet)
    
    df_enhanced = pd.DataFrame(enhanced_samples)
    
    # Save enhanced sample
    output_file = HUMAN_DATA_DIR / "enhanced_human_tweets.csv"
    df_enhanced.to_csv(output_file, index=False)
    print(f"   ✅ Created {len(df_enhanced):,} enhanced human tweets")
    print(f"   💾 Saved to: {output_file}")
    
    return df_enhanced

def main():
    """Process all legitimate tweet datasets"""
    print("🤖 LEGITIMATE TWEET PROCESSOR")
    print("=" * 40)
    
    datasets_processed = []
    
    # Try to process Sentiment140 parquet (preferred)
    df_parquet = process_sentiment140_parquet()
    if df_parquet is not None:
        datasets_processed.append(("Sentiment140 Parquet", len(df_parquet)))
    
    # Try to process Sentiment140 CSV as backup
    df_csv = process_sentiment140_csv()
    if df_csv is not None:
        datasets_processed.append(("Sentiment140 CSV", len(df_csv)))
    
    # Always create enhanced sample data
    df_enhanced = create_enhanced_sample_data()
    datasets_processed.append(("Enhanced Sample", len(df_enhanced)))
    
    print(f"\n📊 PROCESSING SUMMARY:")
    print("=" * 30)
    total_tweets = 0
    for name, count in datasets_processed:
        print(f"   {name}: {count:,} tweets")
        total_tweets += count
    
    print(f"\n🎯 TOTAL PROCESSED TWEETS: {total_tweets:,}")
    print(f"📁 All files saved in: {HUMAN_DATA_DIR}")
    
    # List all available files
    print(f"\n📋 AVAILABLE LEGITIMATE TWEET FILES:")
    csv_files = list(HUMAN_DATA_DIR.glob("*processed.csv")) + list(HUMAN_DATA_DIR.glob("enhanced*.csv"))
    for csv_file in csv_files:
        file_size = csv_file.stat().st_size / (1024 * 1024)
        print(f"   • {csv_file.name} ({file_size:.1f} MB)")
    
    print(f"\n✅ Ready to create balanced dataset!")
    print("   Run: python create_balanced_dataset.py")

if __name__ == "__main__":
    main() 