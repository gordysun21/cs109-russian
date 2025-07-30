"""
Central Limit Theorem Applications in Bot Detection - Simple Demo

This demonstrates the key ways CLT applies to our bot detection project
using synthetic data that mirrors our real dataset patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set style and random seed
plt.style.use('seaborn-v0_8')
np.random.seed(42)

def demonstrate_clt_feature_sampling():
    """
    CLT Application 1: Sampling Distribution of Feature Means
    
    Shows how follower counts from bot accounts follow CLT when we take sample means
    """
    print("📊 CLT APPLICATION 1: SAMPLING DISTRIBUTION OF FEATURE MEANS")
    print("=" * 70)
    
    # Create synthetic bot follower data (mimics our real Russian bot data)
    # Russian bots typically have skewed distributions
    n_population = 50000
    bot_followers = np.random.lognormal(mean=6, sigma=1.5, size=n_population)
    
    pop_mean = np.mean(bot_followers)
    pop_std = np.std(bot_followers)
    
    print(f"🎯 Bot Follower Counts Analysis:")
    print(f"   Population mean (μ): {pop_mean:.0f}")
    print(f"   Population std (σ): {pop_std:.0f}")
    print(f"   Population size: {n_population:,}")
    
    # Demonstrate CLT with different sample sizes
    sample_sizes = [5, 15, 30, 100]
    n_samples = 1000
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Original population (highly skewed)
    axes[0].hist(bot_followers, bins=100, alpha=0.7, color='lightblue', density=True)
    axes[0].set_title('Original Bot Follower Distribution\n(Highly Skewed - NOT Normal)')
    axes[0].set_xlabel('Followers')
    axes[0].set_ylabel('Density')
    axes[0].axvline(pop_mean, color='red', linestyle='--', label=f'μ = {pop_mean:.0f}')
    axes[0].legend()
    axes[0].set_xlim(0, 10000)
    
    # Sample means for different sample sizes
    for i, n in enumerate(sample_sizes):
        sample_means = []
        for _ in range(n_samples):
            sample = np.random.choice(bot_followers, size=n, replace=True)
            sample_means.append(np.mean(sample))
        
        sample_means = np.array(sample_means)
        
        # CLT prediction: mean = μ, std = σ/√n
        theoretical_mean = pop_mean
        theoretical_std = pop_std / np.sqrt(n)
        
        # Plot sample means distribution
        ax = axes[i + 1]
        ax.hist(sample_means, bins=50, alpha=0.7, color='lightgreen', density=True)
        
        # Overlay theoretical normal
        x = np.linspace(sample_means.min(), sample_means.max(), 100)
        normal_curve = stats.norm.pdf(x, theoretical_mean, theoretical_std)
        ax.plot(x, normal_curve, 'r-', linewidth=3, label='CLT Prediction')
        
        ax.set_title(f'Sample Means (n={n})\\nBecoming More Normal!')
        ax.set_xlabel('Sample Mean Followers')
        ax.set_ylabel('Density')
        ax.legend()
        
        # Print comparison
        actual_mean = np.mean(sample_means)
        actual_std = np.std(sample_means)
        
        print(f"\\n   Sample size n = {n}:")
        print(f"     CLT predicts: μ = {theoretical_mean:.0f}, σ = {theoretical_std:.0f}")
        print(f"     Actual result: μ = {actual_mean:.0f}, σ = {actual_std:.0f}")
        print(f"     Error: {abs(theoretical_std - actual_std):.1f}")
    
    plt.tight_layout()
    plt.savefig('clt_demo_sampling.png', dpi=300, bbox_inches='tight')
    print(f"\\n   📊 Saved to 'clt_demo_sampling.png'")

def demonstrate_clt_model_confidence():
    """
    CLT Application 2: Model Performance Confidence Intervals
    
    Shows how CLT lets us create confidence intervals for model accuracy
    """
    print("\\n📈 CLT APPLICATION 2: MODEL PERFORMANCE CONFIDENCE INTERVALS")
    print("=" * 70)
    
    # Simulate model predictions on test batches
    true_accuracy = 0.92  # Our model's true accuracy
    
    print(f"🎯 Simulating model with true accuracy = {true_accuracy:.1%}")
    
    # Different test set sizes
    test_sizes = [50, 100, 500, 1000]
    n_trials = 1000
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, test_size in enumerate(test_sizes):
        accuracies = []
        
        # Simulate many test runs
        for _ in range(n_trials):
            # Simulate predictions (binomial: each prediction correct with prob = true_accuracy)
            correct_predictions = np.random.binomial(test_size, true_accuracy)
            accuracy = correct_predictions / test_size
            accuracies.append(accuracy)
        
        accuracies = np.array(accuracies)
        
        # CLT prediction for binomial proportion
        theoretical_mean = true_accuracy
        theoretical_std = np.sqrt(true_accuracy * (1 - true_accuracy) / test_size)
        
        # Plot distribution
        ax = axes[i]
        ax.hist(accuracies, bins=30, alpha=0.7, color='lightcoral', density=True)
        
        # Overlay normal approximation
        x = np.linspace(accuracies.min(), accuracies.max(), 100)
        normal_curve = stats.norm.pdf(x, theoretical_mean, theoretical_std)
        ax.plot(x, normal_curve, 'b-', linewidth=3, label='CLT Normal Approx')
        
        # 95% confidence interval
        ci_lower = np.percentile(accuracies, 2.5)
        ci_upper = np.percentile(accuracies, 97.5)
        
        ax.axvline(ci_lower, color='red', linestyle='--', alpha=0.7)
        ax.axvline(ci_upper, color='red', linestyle='--', alpha=0.7)
        ax.axvline(theoretical_mean, color='blue', linestyle='-', alpha=0.8)
        
        ax.set_title(f'Test Set Size = {test_size}\\n95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
        ax.set_xlabel('Observed Accuracy')
        ax.set_ylabel('Density')
        ax.legend()
        
        print(f"\\n   Test size {test_size}:")
        print(f"     CLT std error: {theoretical_std:.4f}")
        print(f"     Actual std: {np.std(accuracies):.4f}")
        print(f"     95% CI width: {ci_upper - ci_lower:.4f}")
    
    plt.tight_layout()
    plt.savefig('clt_demo_confidence.png', dpi=300, bbox_inches='tight')
    print(f"\\n   📊 Saved to 'clt_demo_confidence.png'")

def demonstrate_clt_hypothesis_testing():
    """
    CLT Application 3: Hypothesis Testing Between Bot Types
    
    Shows how CLT enables valid t-tests comparing different bot groups
    """
    print("\\n🔬 CLT APPLICATION 3: HYPOTHESIS TESTING BETWEEN BOT GROUPS")
    print("=" * 70)
    
    print("🎯 Testing: Do Russian bots vs Human accounts have different follower patterns?")
    
    # Simulate bot vs human follower data
    n_each = 2000
    
    # Russian bots: lower followers, higher variance
    russian_followers = np.random.lognormal(mean=5.5, sigma=1.8, size=n_each)
    
    # Human accounts: higher followers, lower variance  
    human_followers = np.random.lognormal(mean=6.5, sigma=1.2, size=n_each)
    
    # CLT enables t-test even though original distributions are log-normal (not normal)
    t_stat, p_value = stats.ttest_ind(russian_followers, human_followers)
    
    # Effect size
    pooled_std = np.sqrt(((n_each-1)*np.var(russian_followers) + (n_each-1)*np.var(human_followers)) / (2*n_each-2))
    cohens_d = (np.mean(russian_followers) - np.mean(human_followers)) / pooled_std
    
    print(f"\\n   📊 Results:")
    print(f"     Russian bot mean: {np.mean(russian_followers):.0f} followers")
    print(f"     Human mean: {np.mean(human_followers):.0f} followers")
    print(f"     t-statistic: {t_stat:.4f}")
    print(f"     p-value: {p_value:.2e}")
    print(f"     Effect size (Cohen's d): {cohens_d:.4f}")
    print(f"     Significant difference: {'YES' if p_value < 0.05 else 'NO'}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original distributions
    ax1.hist(russian_followers, bins=50, alpha=0.6, label='Russian Bots', color='red', density=True)
    ax1.hist(human_followers, bins=50, alpha=0.6, label='Human Accounts', color='blue', density=True)
    ax1.axvline(np.mean(russian_followers), color='red', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(human_followers), color='blue', linestyle='--', linewidth=2)
    ax1.set_title('Original Distributions\\n(Non-normal, but CLT still applies!)')
    ax1.set_xlabel('Followers')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.set_xlim(0, 5000)
    
    # Sampling distributions of means (what CLT gives us)
    sample_means_russian = []
    sample_means_human = []
    
    for _ in range(1000):
        russian_sample = np.random.choice(russian_followers, 50)
        human_sample = np.random.choice(human_followers, 50)
        sample_means_russian.append(np.mean(russian_sample))
        sample_means_human.append(np.mean(human_sample))
    
    ax2.hist(sample_means_russian, bins=30, alpha=0.6, label='Russian Sample Means', color='red', density=True)
    ax2.hist(sample_means_human, bins=30, alpha=0.6, label='Human Sample Means', color='blue', density=True)
    ax2.set_title('Sample Means Distributions\\n(Normal by CLT - enables t-test!)')
    ax2.set_xlabel('Sample Mean Followers')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('clt_demo_testing.png', dpi=300, bbox_inches='tight')
    print(f"\\n   📊 Saved to 'clt_demo_testing.png'")

def demonstrate_clt_prediction_batches():
    """
    CLT Application 4: Batch Prediction Analysis
    
    Shows how CLT applies when analyzing batches of predictions
    """
    print("\\n🤖 CLT APPLICATION 4: BATCH PREDICTION ANALYSIS")
    print("=" * 70)
    
    print("🎯 Analyzing how prediction scores behave in batches")
    
    # Simulate individual prediction probabilities (bimodal: clear bots vs clear humans)
    n_predictions = 10000
    bot_probs = np.concatenate([
        np.random.beta(8, 2, n_predictions//2),    # Clear bots (high probability)
        np.random.beta(2, 8, n_predictions//2)     # Clear humans (low probability)
    ])
    
    print(f"   Individual predictions: {len(bot_probs):,}")
    print(f"   Mean prediction: {np.mean(bot_probs):.3f}")
    print(f"   Distribution: Bimodal (not normal)")
    
    # Show how batch means become normal
    batch_sizes = [5, 20, 50, 100]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Original distribution (bimodal)
    axes[0].hist(bot_probs, bins=50, alpha=0.7, color='lightblue', density=True)
    axes[0].set_title('Individual Prediction Scores\\n(Bimodal - NOT Normal)')
    axes[0].set_xlabel('Bot Probability')
    axes[0].set_ylabel('Density')
    
    for i, batch_size in enumerate(batch_sizes):
        batch_means = []
        n_batches = 1000
        
        for _ in range(n_batches):
            batch = np.random.choice(bot_probs, batch_size)
            batch_means.append(np.mean(batch))
        
        batch_means = np.array(batch_means)
        
        # CLT prediction
        theoretical_mean = np.mean(bot_probs)
        theoretical_std = np.std(bot_probs) / np.sqrt(batch_size)
        
        # Plot
        ax = axes[i + 1]
        ax.hist(batch_means, bins=30, alpha=0.7, color='lightgreen', density=True)
        
        # Normal overlay
        x = np.linspace(batch_means.min(), batch_means.max(), 100)
        normal_curve = stats.norm.pdf(x, theoretical_mean, theoretical_std)
        ax.plot(x, normal_curve, 'r-', linewidth=3, label='CLT Prediction')
        
        ax.set_title(f'Batch Means (size={batch_size})\\nBecoming Normal!')
        ax.set_xlabel('Batch Mean Probability')
        ax.set_ylabel('Density')
        ax.legend()
        
        # Normality test
        _, p_value = stats.shapiro(batch_means[:50])  # Test subset
        
        print(f"\\n   Batch size {batch_size}:")
        print(f"     Normality test p-value: {p_value:.4f}")
        print(f"     {'NORMAL' if p_value > 0.05 else 'not quite normal yet'}")
    
    plt.tight_layout()
    plt.savefig('clt_demo_batches.png', dpi=300, bbox_inches='tight')
    print(f"\\n   📊 Saved to 'clt_demo_batches.png'")

def summarize_clt_importance():
    """Explain why CLT matters for bot detection"""
    print("\\n🎓 WHY CENTRAL LIMIT THEOREM MATTERS FOR BOT DETECTION")
    print("=" * 70)
    
    print("""
    The Central Limit Theorem is CRUCIAL for our bot detection project because:
    
    1. 📊 STATISTICAL INFERENCE
       • Even if follower counts are skewed, sample means are normal
       • This lets us calculate confidence intervals and p-values
       • We can make probabilistic statements about our results
    
    2. 🤖 MODEL EVALUATION  
       • CLT enables bootstrap confidence intervals for accuracy
       • We can say "Model accuracy is 92% ± 2%" with confidence
       • Helps compare different models statistically
    
    3. 🔬 HYPOTHESIS TESTING
       • CLT justifies t-tests comparing bot vs human features
       • Even with non-normal data, we can test differences
       • Essential for validating that bots behave differently
    
    4. 📈 PREDICTION CONFIDENCE
       • When predicting on batches, CLT gives us error bars
       • Helps detect when model performance is changing
       • Critical for production monitoring
    
    5. 🎯 SAMPLE SIZE PLANNING
       • CLT tells us how big samples need to be
       • Smaller samples → larger error bars
       • Helps plan data collection and experiments
    
    KEY INSIGHT: CLT transforms messy, real-world data into nice normal
    distributions that we can analyze with standard statistical tools!
    """)

def main():
    """Run all CLT demonstrations"""
    print("📐 CENTRAL LIMIT THEOREM IN BOT DETECTION")
    print("=" * 50)
    print("Demonstrating how CLT enables our statistical analysis\\n")
    
    demonstrate_clt_feature_sampling()
    demonstrate_clt_model_confidence()
    demonstrate_clt_hypothesis_testing()
    demonstrate_clt_prediction_batches()
    summarize_clt_importance()
    
    print("\\n✅ CLT analysis complete!")
    print("\\n📁 Generated files:")
    print("   • clt_demo_sampling.png")
    print("   • clt_demo_confidence.png")
    print("   • clt_demo_testing.png")
    print("   • clt_demo_batches.png")

if __name__ == "__main__":
    main() 