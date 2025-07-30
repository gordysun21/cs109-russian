"""
Bot Detection Modeling Analysis: Challenges and Solutions

This script analyzes the viability of using the Russian troll tweets dataset
for building bot detection models and proposes several approaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import re

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

print("🤖 Bot Detection Modeling Analysis")
print("=" * 50)

# Load data for analysis
data_path = Path("data/russian_troll_tweets/IRAhandle_tweets_1.csv")
if not data_path.exists():
    print("❌ Data file not found. Please run copy_dataset.py first to download the data.")
    exit(1)

df = pd.read_csv(data_path)

print("\n❌ CRITICAL LIMITATION: Single-Class Dataset")
print("-" * 40)
print("This dataset contains ONLY Russian bot/troll tweets.")
print("For supervised learning, you need both:")
print("  ✓ Positive examples (bot tweets) - ✅ WE HAVE THIS")
print("  ✗ Negative examples (human tweets) - ❌ WE DON'T HAVE THIS")
print()

print("📊 What We CAN'T Do Directly:")
print("  1. Train a binary classifier (bot vs human)")
print("  2. Generate probability estimates for 'bot-ness'")
print("  3. Validate model performance without human tweets")
print("  4. Create balanced training/test sets")
print()

print("🎯 What We CAN Do:")
print("  1. Feature engineering for bot characteristics")
print("  2. Unsupervised anomaly detection")
print("  3. Pattern analysis for bot behavior")
print("  4. Combine with other datasets")
print()

print("🔧 SOLUTION STRATEGIES:")
print("=" * 30)

print("\n1️⃣ COMBINE WITH LEGITIMATE TWEET DATASETS")
print("   Suggested datasets to combine:")
print("   • Twitter API samples of random tweets")
print("   • Kaggle datasets with verified human tweets")
print("   • Academic datasets with labeled bot/human tweets")
print("   • Sentiment140 dataset (1.6M tweets)")

print("\n2️⃣ UNSUPERVISED APPROACHES")
print("   • One-Class SVM (treat bots as 'normal', detect outliers)")
print("   • Isolation Forest for anomaly detection")
print("   • Clustering to identify bot behavior patterns")
print("   • Autoencoders for learning bot representations")

print("\n3️⃣ FEATURE ENGINEERING FOCUS")
print("   Use this dataset to understand bot characteristics:")

# Analyze bot characteristics
print("\n📈 Bot Behavior Patterns Found:")

# Account metrics analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Followers vs Following ratio
df['follower_following_ratio'] = df['followers'] / (df['following'] + 1)
axes[0,0].hist(np.log10(df['follower_following_ratio'] + 1), bins=50, alpha=0.7)
axes[0,0].set_title('Log10(Follower/Following Ratio + 1)')
axes[0,0].set_xlabel('Log10 Ratio')

# Updates distribution
axes[0,1].hist(np.log10(df['updates'] + 1), bins=50, alpha=0.7, color='orange')
axes[0,1].set_title('Log10(Number of Updates)')
axes[0,1].set_xlabel('Log10 Updates')

# Account types
df['account_category'].value_counts().plot(kind='bar', ax=axes[1,0])
axes[1,0].set_title('Bot Account Categories')
axes[1,0].tick_params(axis='x', rotation=45)

# Retweet percentage
retweet_pct = df['retweet'].mean() * 100
axes[1,1].bar(['Original Tweets', 'Retweets'], 
              [100-retweet_pct, retweet_pct], 
              color=['skyblue', 'lightcoral'])
axes[1,1].set_title(f'Tweet Types (Retweets: {retweet_pct:.1f}%)')
axes[1,1].set_ylabel('Percentage')

plt.tight_layout()
plt.savefig('bot_behavior_analysis.png', dpi=300, bbox_inches='tight')
print("📊 Bot behavior analysis chart saved as 'bot_behavior_analysis.png'")

# Text analysis
print("\n📝 Text Pattern Analysis:")
content_lengths = df['content'].str.len()
print(f"Average tweet length: {content_lengths.mean():.1f} characters")
print(f"Median tweet length: {content_lengths.median():.1f} characters")

# URL pattern
url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
df['has_url'] = df['content'].str.contains(url_pattern, regex=True)
print(f"Tweets with URLs: {df['has_url'].mean()*100:.1f}%")

# Hashtag pattern
df['hashtag_count'] = df['content'].str.count(r'#\w+')
print(f"Average hashtags per tweet: {df['hashtag_count'].mean():.2f}")

# Mention pattern
df['mention_count'] = df['content'].str.count(r'@\w+')
print(f"Average mentions per tweet: {df['mention_count'].mean():.2f}")

print("\n4️⃣ RECOMMENDED MODELING APPROACH")
print("=" * 35)

print("""
STEP 1: Acquire Complementary Data
   • Download legitimate tweet datasets
   • Use Twitter Academic API for random samples
   • Consider using existing bot detection datasets

STEP 2: Feature Engineering (use this dataset)
   • Account age indicators
   • Follower/following ratios
   • Tweet frequency patterns
   • Text characteristics (length, URLs, hashtags)
   • Temporal posting patterns
   • Engagement metrics

STEP 3: Model Development
   • Binary classification (bot vs human)
   • Logistic regression with engineered features
   • Random Forest for feature importance
   • Neural networks for complex patterns

STEP 4: Validation Strategy
   • Cross-validation with balanced datasets
   • Temporal validation (train on old, test on new)
   • Performance metrics: Precision, Recall, F1-score
""")

print("\n5️⃣ ALTERNATIVE: PSEUDO-LABELING APPROACH")
print("=" * 40)
print("""
If you can't get legitimate tweets:
1. Use this bot dataset to train an autoencoder
2. Apply autoencoder to unlabeled tweets
3. Tweets with high reconstruction error = likely human
4. Tweets with low reconstruction error = likely bot
5. Use reconstruction error as 'bot probability'

Note: This is less reliable than supervised learning!
""")

print("\n🎯 CONCLUSION:")
print("While this dataset alone cannot train a bot classifier,")
print("it's EXCELLENT for understanding bot behavior patterns")
print("and engineering features for bot detection systems.")
print("Combine it with legitimate tweet data for best results!") 