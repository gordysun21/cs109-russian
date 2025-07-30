"""
Central Limit Theorem Applications in Bot Detection

This script demonstrates how the Central Limit Theorem applies to various aspects
of our bot detection analysis, including feature distributions, model predictions,
and statistical inference.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Data directories
BALANCED_DATA_DIR = Path("data/balanced_bot_detection")
MODELS_DIR = Path("models")

def load_bot_detection_data():
    """Load our bot detection datasets"""
    print("📥 Loading bot detection data for CLT analysis...")
    
    try:
        # Load the balanced dataset
        balanced_path = BALANCED_DATA_DIR / "balanced_bot_detection_dataset.csv"
        if balanced_path.exists():
            df = pd.read_csv(balanced_path)
            print(f"   ✅ Loaded {len(df):,} samples from balanced dataset")
            return df
        else:
            print("   ❌ Balanced dataset not found. Run create_balanced_dataset.py first.")
            return None
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return None

def demonstrate_clt_feature_distributions(df):
    """
    CLT Application 1: Sampling Distribution of Feature Means
    
    For any population with mean μ and standard deviation σ, the sampling distribution
    of sample means approaches a normal distribution as sample size increases.
    """
    print("\n📊 CLT APPLICATION 1: SAMPLING DISTRIBUTION OF FEATURE MEANS")
    print("=" * 70)
    
    # Select a feature to analyze (follower count)
    feature = 'followers'
    if feature not in df.columns:
        print(f"   ⚠️  Feature '{feature}' not found, skipping this analysis")
        return
    
    population_data = df[feature].dropna()
    pop_mean = population_data.mean()
    pop_std = population_data.std()
    
    print(f"\n🎯 Analyzing feature: {feature}")
    print(f"   Population mean (μ): {pop_mean:.2f}")
    print(f"   Population std (σ): {pop_std:.2f}")
    print(f"   Population size: {len(population_data):,}")
    
    # Demonstrate CLT with different sample sizes
    sample_sizes = [5, 10, 30, 100, 500]
    n_samples = 1000  # Number of samples to draw
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Original population distribution
    axes[0].hist(population_data, bins=50, alpha=0.7, color='lightblue', density=True)
    axes[0].set_title(f'Original Population Distribution\n{feature}')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel('Density')
    axes[0].axvline(pop_mean, color='red', linestyle='--', label=f'μ = {pop_mean:.2f}')
    axes[0].legend()
    
    # Sample means for different sample sizes
    for i, n in enumerate(sample_sizes):
        # Draw many samples of size n and calculate their means
        sample_means = []
        for _ in range(n_samples):
            sample = np.random.choice(population_data, size=n, replace=True)
            sample_means.append(np.mean(sample))
        
        sample_means = np.array(sample_means)
        
        # Calculate theoretical values
        theoretical_mean = pop_mean
        theoretical_std = pop_std / np.sqrt(n)  # Standard Error of the Mean
        
        # Plot distribution of sample means
        ax = axes[i + 1]
        ax.hist(sample_means, bins=50, alpha=0.7, color='lightgreen', density=True)
        
        # Overlay theoretical normal distribution
        x = np.linspace(sample_means.min(), sample_means.max(), 100)
        theoretical_dist = stats.norm.pdf(x, theoretical_mean, theoretical_std)
        ax.plot(x, theoretical_dist, 'r-', linewidth=2, label='Theoretical Normal')
        
        ax.set_title(f'Sample Means Distribution\nn = {n}')
        ax.set_xlabel(f'Sample Mean of {feature}')
        ax.set_ylabel('Density')
        ax.axvline(theoretical_mean, color='red', linestyle='--', alpha=0.7)
        ax.legend()
        
        # Calculate actual vs theoretical statistics
        actual_mean = np.mean(sample_means)
        actual_std = np.std(sample_means)
        
        print(f"\n   Sample size n = {n}:")
        print(f"     Theoretical mean: {theoretical_mean:.2f}, Actual: {actual_mean:.2f}")
        print(f"     Theoretical SEM: {theoretical_std:.2f}, Actual: {actual_std:.2f}")
        print(f"     Difference: {abs(theoretical_std - actual_std):.4f}")
    
    plt.tight_layout()
    plt.savefig('clt_feature_distributions.png', dpi=300, bbox_inches='tight')
    print(f"\n   📊 Saved visualization to 'clt_feature_distributions.png'")

def demonstrate_clt_model_predictions(df):
    """
    CLT Application 2: Distribution of Model Prediction Scores
    
    When we make predictions on batches of tweets, the CLT tells us that
    the distribution of batch prediction scores will approach normality.
    """
    print("\n🤖 CLT APPLICATION 2: MODEL PREDICTION DISTRIBUTIONS")
    print("=" * 70)
    
    try:
        # Load trained model
        lr_model = joblib.load(MODELS_DIR / "logistic_regression_bot_detector.pkl")
        scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")
        
        with open(MODELS_DIR / "feature_columns.txt", 'r') as f:
            feature_cols = [line.strip() for line in f.readlines()]
        
        print("   ✅ Loaded trained model")
        
    except FileNotFoundError:
        print("   ❌ Trained models not found. Run train_bot_classifier.py first.")
        return
    
    # Prepare features
    available_features = [col for col in feature_cols if col in df.columns]
    if len(available_features) < len(feature_cols) * 0.8:  # If we're missing too many features
        print("   ⚠️  Too many features missing, skipping prediction analysis")
        return
    
    X = df[available_features].fillna(0)
    y = df['bot_label'] if 'bot_label' in df.columns else np.random.choice([0, 1], len(df))
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get prediction probabilities
    prediction_probs = lr_model.predict_proba(X_scaled)[:, 1]
    
    print(f"\n🎯 Analyzing {len(prediction_probs):,} prediction scores")
    print(f"   Mean prediction score: {np.mean(prediction_probs):.4f}")
    print(f"   Std prediction score: {np.std(prediction_probs):.4f}")
    
    # Demonstrate CLT with batch predictions
    batch_sizes = [10, 30, 100, 500]
    n_batches = 1000
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, batch_size in enumerate(batch_sizes):
        # Create many batches and calculate mean prediction for each batch
        batch_means = []
        for _ in range(n_batches):
            batch_indices = np.random.choice(len(prediction_probs), size=batch_size, replace=True)
            batch_predictions = prediction_probs[batch_indices]
            batch_means.append(np.mean(batch_predictions))
        
        batch_means = np.array(batch_means)
        
        # Plot distribution of batch means
        ax = axes[i]
        ax.hist(batch_means, bins=50, alpha=0.7, color='lightcoral', density=True)
        
        # Overlay normal distribution
        mean_of_means = np.mean(batch_means)
        std_of_means = np.std(batch_means)
        
        x = np.linspace(batch_means.min(), batch_means.max(), 100)
        normal_dist = stats.norm.pdf(x, mean_of_means, std_of_means)
        ax.plot(x, normal_dist, 'b-', linewidth=2, label='Fitted Normal')
        
        ax.set_title(f'Distribution of Batch Mean Predictions\nBatch Size = {batch_size}')
        ax.set_xlabel('Mean Prediction Score')
        ax.set_ylabel('Density')
        ax.legend()
        
        # Test normality
        _, p_value = stats.shapiro(batch_means[:50])  # Shapiro-Wilk test on subset
        print(f"\n   Batch size {batch_size}:")
        print(f"     Mean of batch means: {mean_of_means:.4f}")
        print(f"     Std of batch means: {std_of_means:.4f}")
        print(f"     Normality test p-value: {p_value:.4f}")
    
    plt.tight_layout()
    plt.savefig('clt_prediction_distributions.png', dpi=300, bbox_inches='tight')
    print(f"\n   📊 Saved visualization to 'clt_prediction_distributions.png'")

def demonstrate_clt_confidence_intervals(df):
    """
    CLT Application 3: Confidence Intervals for Model Performance
    
    CLT allows us to create confidence intervals for accuracy and other metrics
    using bootstrap sampling.
    """
    print("\n📈 CLT APPLICATION 3: CONFIDENCE INTERVALS FOR MODEL PERFORMANCE")
    print("=" * 70)
    
    if 'bot_label' not in df.columns:
        print("   ⚠️  No bot_label column found, skipping confidence interval analysis")
        return
    
    try:
        # Load model
        lr_model = joblib.load(MODELS_DIR / "logistic_regression_bot_detector.pkl")
        scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")
        
        with open(MODELS_DIR / "feature_columns.txt", 'r') as f:
            feature_cols = [line.strip() for line in f.readlines()]
        
    except FileNotFoundError:
        print("   ❌ Trained models not found. Run train_bot_classifier.py first.")
        return
    
    # Prepare data
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features].fillna(0)
    y = df['bot_label']
    
    X_scaled = scaler.transform(X)
    y_pred = lr_model.predict(X_scaled)
    
    print(f"\n🎯 Bootstrap Analysis for Model Accuracy")
    print(f"   Dataset size: {len(y):,}")
    
    # Bootstrap sampling for confidence intervals
    n_bootstrap = 1000
    bootstrap_accuracies = []
    
    print("   Performing bootstrap sampling...")
    for i in range(n_bootstrap):
        if i % 200 == 0:
            print(f"     Bootstrap sample {i:,}/{n_bootstrap:,}")
        
        # Bootstrap sample
        indices = np.random.choice(len(y), size=len(y), replace=True)
        y_boot = y.iloc[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate accuracy for this bootstrap sample
        accuracy = accuracy_score(y_boot, y_pred_boot)
        bootstrap_accuracies.append(accuracy)
    
    bootstrap_accuracies = np.array(bootstrap_accuracies)
    
    # Calculate confidence intervals
    confidence_levels = [90, 95, 99]
    
    print(f"\n   Bootstrap Results:")
    print(f"     Mean accuracy: {np.mean(bootstrap_accuracies):.4f}")
    print(f"     Std accuracy: {np.std(bootstrap_accuracies):.4f}")
    
    for conf_level in confidence_levels:
        alpha = (100 - conf_level) / 100
        lower = np.percentile(bootstrap_accuracies, 100 * alpha / 2)
        upper = np.percentile(bootstrap_accuracies, 100 * (1 - alpha / 2))
        
        print(f"     {conf_level}% CI: [{lower:.4f}, {upper:.4f}]")
    
    # Plot bootstrap distribution
    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_accuracies, bins=50, alpha=0.7, color='lightgreen', density=True)
    
    # Overlay normal distribution
    mean_acc = np.mean(bootstrap_accuracies)
    std_acc = np.std(bootstrap_accuracies)
    x = np.linspace(bootstrap_accuracies.min(), bootstrap_accuracies.max(), 100)
    normal_dist = stats.norm.pdf(x, mean_acc, std_acc)
    plt.plot(x, normal_dist, 'r-', linewidth=2, label='Normal Approximation')
    
    # Add confidence interval lines
    ci_95_lower = np.percentile(bootstrap_accuracies, 2.5)
    ci_95_upper = np.percentile(bootstrap_accuracies, 97.5)
    
    plt.axvline(ci_95_lower, color='red', linestyle='--', alpha=0.7, label='95% CI')
    plt.axvline(ci_95_upper, color='red', linestyle='--', alpha=0.7)
    plt.axvline(mean_acc, color='blue', linestyle='-', alpha=0.7, label=f'Mean = {mean_acc:.4f}')
    
    plt.title('Bootstrap Distribution of Model Accuracy\n(Central Limit Theorem in Action)')
    plt.xlabel('Accuracy')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('clt_confidence_intervals.png', dpi=300, bbox_inches='tight')
    print(f"\n   📊 Saved visualization to 'clt_confidence_intervals.png'")

def demonstrate_clt_hypothesis_testing(df):
    """
    CLT Application 4: Hypothesis Testing
    
    CLT enables us to perform statistical tests comparing features between
    Russian bots and human tweets.
    """
    print("\n🔬 CLT APPLICATION 4: HYPOTHESIS TESTING (RUSSIAN vs HUMAN)")
    print("=" * 70)
    
    if 'bot_label' not in df.columns:
        print("   ⚠️  No bot_label column found, skipping hypothesis testing")
        return
    
    # Separate bot and human data
    bot_data = df[df['bot_label'] == 1]
    human_data = df[df['bot_label'] == 0]
    
    print(f"\n🎯 Comparing features between groups:")
    print(f"   Bot tweets: {len(bot_data):,}")
    print(f"   Human tweets: {len(human_data):,}")
    
    # Test several features
    features_to_test = ['followers', 'following', 'updates']
    results = []
    
    for feature in features_to_test:
        if feature not in df.columns:
            continue
        
        bot_values = bot_data[feature].dropna()
        human_values = human_data[feature].dropna()
        
        if len(bot_values) < 30 or len(human_values) < 30:
            continue
        
        # Perform two-sample t-test
        # CLT justifies using t-test even if original distributions aren't normal
        t_stat, p_value = stats.ttest_ind(bot_values, human_values)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(bot_values) - 1) * bot_values.var() + 
                             (len(human_values) - 1) * human_values.var()) / 
                            (len(bot_values) + len(human_values) - 2))
        cohens_d = (bot_values.mean() - human_values.mean()) / pooled_std
        
        results.append({
            'feature': feature,
            'bot_mean': bot_values.mean(),
            'human_mean': human_values.mean(),
            'bot_std': bot_values.std(),
            'human_std': human_values.std(),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        })
        
        print(f"\n   📊 {feature}:")
        print(f"     Bot mean: {bot_values.mean():.2f} ± {bot_values.std():.2f}")
        print(f"     Human mean: {human_values.mean():.2f} ± {human_values.std():.2f}")
        print(f"     t-statistic: {t_stat:.4f}")
        print(f"     p-value: {p_value:.6f}")
        print(f"     Cohen's d: {cohens_d:.4f}")
        print(f"     Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Create comparison plots
    if results:
        fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 6))
        if len(results) == 1:
            axes = [axes]
        
        for i, result in enumerate(results):
            feature = result['feature']
            bot_values = bot_data[feature].dropna()
            human_values = human_data[feature].dropna()
            
            ax = axes[i]
            
            # Plot distributions
            ax.hist(bot_values, bins=30, alpha=0.7, label='Bot', color='red', density=True)
            ax.hist(human_values, bins=30, alpha=0.7, label='Human', color='blue', density=True)
            
            ax.axvline(result['bot_mean'], color='red', linestyle='--', label=f"Bot μ = {result['bot_mean']:.1f}")
            ax.axvline(result['human_mean'], color='blue', linestyle='--', label=f"Human μ = {result['human_mean']:.1f}")
            
            ax.set_title(f"{feature}\np = {result['p_value']:.4f}")
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('clt_hypothesis_testing.png', dpi=300, bbox_inches='tight')
        print(f"\n   📊 Saved comparison plots to 'clt_hypothesis_testing.png'")

def demonstrate_clt_aggregation(df):
    """
    CLT Application 5: Account-Level Aggregation
    
    When aggregating tweet features by account, CLT tells us that the
    distribution of account-level averages will be approximately normal.
    """
    print("\n👤 CLT APPLICATION 5: ACCOUNT-LEVEL FEATURE AGGREGATION")
    print("=" * 70)
    
    if 'author' not in df.columns:
        print("   ⚠️  No author column found, creating synthetic accounts")
        # Create synthetic account groupings
        n_accounts = len(df) // 10  # About 10 tweets per account on average
        df['author'] = np.random.choice([f'user_{i}' for i in range(n_accounts)], len(df))
    
    # Focus on accounts with multiple tweets
    author_counts = df['author'].value_counts()
    multi_tweet_authors = author_counts[author_counts >= 5].index
    
    print(f"\n🎯 Account-level aggregation:")
    print(f"   Total accounts: {len(author_counts):,}")
    print(f"   Accounts with 5+ tweets: {len(multi_tweet_authors):,}")
    
    if len(multi_tweet_authors) < 50:
        print("   ⚠️  Too few accounts with multiple tweets for meaningful analysis")
        return
    
    # Calculate account-level averages
    account_features = []
    feature = 'followers' if 'followers' in df.columns else 'word_count'
    
    for author in multi_tweet_authors[:200]:  # Limit for performance
        author_tweets = df[df['author'] == author]
        if feature in author_tweets.columns:
            avg_feature = author_tweets[feature].mean()
            account_features.append({
                'author': author,
                'avg_feature': avg_feature,
                'tweet_count': len(author_tweets),
                'bot_proportion': author_tweets['bot_label'].mean() if 'bot_label' in df.columns else 0.5
            })
    
    account_df = pd.DataFrame(account_features)
    
    print(f"   Analyzing {len(account_df)} accounts")
    print(f"   Average {feature} per account: {account_df['avg_feature'].mean():.2f}")
    print(f"   Std of account averages: {account_df['avg_feature'].std():.2f}")
    
    # Test normality of account-level averages
    _, p_value = stats.shapiro(account_df['avg_feature'][:50])  # Test subset
    print(f"   Normality test p-value: {p_value:.4f}")
    
    # Plot individual tweets vs account averages
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Individual tweet distribution
    if feature in df.columns:
        tweet_values = df[feature].dropna()
        ax1.hist(tweet_values, bins=50, alpha=0.7, color='lightblue', density=True)
        ax1.set_title(f'Individual Tweet {feature}\n(Original Distribution)')
        ax1.set_xlabel(feature)
        ax1.set_ylabel('Density')
    
    # Account average distribution
    ax2.hist(account_df['avg_feature'], bins=30, alpha=0.7, color='lightgreen', density=True)
    
    # Overlay normal distribution
    mean_avg = account_df['avg_feature'].mean()
    std_avg = account_df['avg_feature'].std()
    x = np.linspace(account_df['avg_feature'].min(), account_df['avg_feature'].max(), 100)
    normal_dist = stats.norm.pdf(x, mean_avg, std_avg)
    ax2.plot(x, normal_dist, 'r-', linewidth=2, label='Normal Approximation')
    
    ax2.set_title(f'Account Average {feature}\n(CLT: More Normal Distribution)')
    ax2.set_xlabel(f'Average {feature}')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('clt_account_aggregation.png', dpi=300, bbox_inches='tight')
    print(f"\n   📊 Saved visualization to 'clt_account_aggregation.png'")

def main():
    """Main function to demonstrate CLT applications"""
    print("📐 CENTRAL LIMIT THEOREM APPLICATIONS IN BOT DETECTION")
    print("=" * 80)
    print("Demonstrating how CLT applies to various aspects of our analysis")
    print()
    
    # Load data
    df = load_bot_detection_data()
    if df is None:
        print("❌ Could not load data. Please run the previous scripts first.")
        return
    
    print(f"🎯 Dataset loaded: {len(df):,} samples")
    
    try:
        # Demonstrate different CLT applications
        demonstrate_clt_feature_distributions(df)
        demonstrate_clt_model_predictions(df)
        demonstrate_clt_confidence_intervals(df)
        demonstrate_clt_hypothesis_testing(df)
        demonstrate_clt_aggregation(df)
        
        print(f"\n✅ SUCCESS! Central Limit Theorem analysis complete")
        print(f"\n🎓 KEY CLT INSIGHTS:")
        print(f"   1. Feature sample means → Normal distribution (enables statistical inference)")
        print(f"   2. Model prediction batches → Normal distribution (prediction confidence)")
        print(f"   3. Bootstrap accuracy → Normal distribution (confidence intervals)")
        print(f"   4. Large samples → Valid t-tests (hypothesis testing)")
        print(f"   5. Account aggregations → More normal distributions (account-level analysis)")
        
        print(f"\n📁 Generated visualizations:")
        print(f"   • clt_feature_distributions.png")
        print(f"   • clt_prediction_distributions.png") 
        print(f"   • clt_confidence_intervals.png")
        print(f"   • clt_hypothesis_testing.png")
        print(f"   • clt_account_aggregation.png")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("Make sure you have run the previous bot detection scripts first.")

if __name__ == "__main__":
    main() 