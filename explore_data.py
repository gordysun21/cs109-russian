import pandas as pd
import numpy as np
from pathlib import Path

# Load and explore the first dataset file
print("🔍 Exploring Russian Troll Tweets Dataset Structure\n")

# Load the first file to understand structure
data_path = Path("data/russian_troll_tweets/IRAhandle_tweets_1.csv")
if not data_path.exists():
    print("❌ Data file not found. Please run copy_dataset.py first to download the data.")
    exit(1)

df1 = pd.read_csv(data_path)

print("📊 Dataset Overview:")
print(f"Shape of first file: {df1.shape}")
print(f"Columns: {list(df1.columns)}")
print()

print("📋 Column Information:")
print(df1.info())
print()

print("🔍 First few rows:")
print(df1.head())
print()

print("📈 Data Summary:")
print(df1.describe(include='all'))
print()

# Check for unique values in key categorical columns
categorical_cols = ['account_type', 'account_category', 'language', 'region']
for col in categorical_cols:
    if col in df1.columns:
        print(f"🏷️  Unique values in '{col}': {df1[col].nunique()}")
        print(f"   Values: {df1[col].value_counts().head().to_dict()}")
        print()

# Sample some tweet content
print("📝 Sample tweet content:")
if 'content' in df1.columns:
    for i, tweet in enumerate(df1['content'].dropna().head(3)):
        print(f"Tweet {i+1}: {tweet[:200]}...")
        print()

# Check date range
if 'publish_date' in df1.columns:
    df1['publish_date'] = pd.to_datetime(df1['publish_date'])
    print(f"📅 Date range: {df1['publish_date'].min()} to {df1['publish_date'].max()}")
    print()

print("⚠️  Key Limitation: This dataset contains ONLY bot/troll tweets")
print("   - No legitimate human tweets for comparison")
print("   - Cannot train a binary classifier directly")
print("   - Need additional data for supervised learning") 