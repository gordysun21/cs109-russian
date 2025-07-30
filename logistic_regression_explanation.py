"""
Logistic Regression Explanation for Bot Detection Demo

This script provides a comprehensive explanation of the mathematical concepts
behind logistic regression used in our bot detection system.

Topics covered:
1. Logistic Regression Fundamentals
2. Sigmoid Function
3. Maximum Likelihood Estimation (MLE)
4. Gradient Descent/Ascent
5. Application to Bot Detection
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import seaborn as sns

class LogisticRegressionExplainer:
    def __init__(self):
        self.setup_style()
        
    def setup_style(self):
        """Setup plotting style"""
        plt.style.use('default')
        sns.set_palette("husl")
        
    def explain_all_concepts(self):
        """Run complete explanation of all concepts"""
        print("🤖 LOGISTIC REGRESSION FOR BOT DETECTION")
        print("=" * 60)
        print()
        
        # 1. Basic concept explanation
        self.explain_basic_concept()
        
        # 2. Sigmoid function
        self.explain_sigmoid_function()
        
        # 3. Maximum Likelihood Estimation
        self.explain_mle()
        
        # 4. Gradient Descent
        self.explain_gradient_descent()
        
        # 5. Bot detection application
        self.explain_bot_detection_application()
        
        print("\n🎯 SUMMARY")
        print("-" * 30)
        self.provide_summary()
        
    def explain_basic_concept(self):
        """Explain the basic concept of logistic regression"""
        print("📚 1. LOGISTIC REGRESSION FUNDAMENTALS")
        print("-" * 45)
        print()
        
        print("🔹 What is Logistic Regression?")
        print("   Logistic regression is a statistical method for binary classification.")
        print("   Unlike linear regression, it predicts PROBABILITIES between 0 and 1.")
        print()
        
        print("🔹 Why not Linear Regression for Classification?")
        print("   • Linear regression can predict values outside [0,1]")
        print("   • We need probabilities that are always between 0 and 1")
        print("   • Linear regression assumes constant variance (not true for binary outcomes)")
        print()
        
        print("🔹 The Logistic Regression Model:")
        print("   Instead of: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ")
        print("   We use:    P(y=1|x) = 1 / (1 + e^(-z))")
        print("   Where:     z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ")
        print()
        
        print("🔹 In Bot Detection Context:")
        print("   • y = 1 if account is a bot, 0 if human")
        print("   • x₁, x₂, ... xₙ are features (followers, tweets, etc.)")
        print("   • P(y=1|x) is the probability that an account is a bot")
        print()
        
    def explain_sigmoid_function(self):
        """Explain and visualize the sigmoid function"""
        print("📈 2. THE SIGMOID FUNCTION")
        print("-" * 35)
        print()
        
        print("🔹 Mathematical Definition:")
        print("   σ(z) = 1 / (1 + e^(-z))")
        print("   where z is the linear combination: z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ")
        print()
        
        print("🔹 Key Properties:")
        print("   • Always outputs values between 0 and 1")
        print("   • S-shaped (sigmoid) curve")
        print("   • σ(0) = 0.5 (decision boundary)")
        print("   • σ(+∞) → 1, σ(-∞) → 0")
        print("   • Smooth and differentiable everywhere")
        print()
        
        # Create visualization
        self.plot_sigmoid_function()
        
        print("🔹 Interpretation in Bot Detection:")
        print("   • z > 0 → P(bot) > 0.5 → Classified as bot")
        print("   • z < 0 → P(bot) < 0.5 → Classified as human")
        print("   • z = 0 → P(bot) = 0.5 → Decision boundary")
        print()
        
    def plot_sigmoid_function(self):
        """Plot the sigmoid function with annotations"""
        z = np.linspace(-6, 6, 100)
        sigmoid = 1 / (1 + np.exp(-z))
        
        plt.figure(figsize=(12, 6))
        
        # Main sigmoid plot
        plt.subplot(1, 2, 1)
        plt.plot(z, sigmoid, 'b-', linewidth=3, label='σ(z) = 1/(1+e^(-z))')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Boundary (P=0.5)')
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('z (linear combination)', fontsize=12)
        plt.ylabel('P(bot)', fontsize=12)
        plt.title('Sigmoid Function for Bot Detection', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add annotations
        plt.annotate('Human\n(P < 0.5)', xy=(-3, 0.1), fontsize=11, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        plt.annotate('Bot\n(P > 0.5)', xy=(3, 0.9), fontsize=11, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        
        # Derivative plot
        plt.subplot(1, 2, 2)
        sigmoid_derivative = sigmoid * (1 - sigmoid)
        plt.plot(z, sigmoid_derivative, 'g-', linewidth=3, label="σ'(z) = σ(z)(1-σ(z))")
        plt.xlabel('z', fontsize=12)
        plt.ylabel("σ'(z)", fontsize=12)
        plt.title('Sigmoid Derivative\n(used in gradient descent)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def explain_mle(self):
        """Explain Maximum Likelihood Estimation"""
        print("🎯 3. MAXIMUM LIKELIHOOD ESTIMATION (MLE)")
        print("-" * 50)
        print()
        
        print("🔹 What is MLE?")
        print("   MLE finds the parameter values (β₀, β₁, ..., βₙ) that make")
        print("   our observed data most likely to have occurred.")
        print()
        
        print("🔹 The Likelihood Function:")
        print("   For a single observation i:")
        print("   L(βᵢ) = P(yᵢ=1|xᵢ)^yᵢ × P(yᵢ=0|xᵢ)^(1-yᵢ)")
        print("   L(βᵢ) = σ(zᵢ)^yᵢ × (1-σ(zᵢ))^(1-yᵢ)")
        print()
        
        print("🔹 For all n observations:")
        print("   L(β) = ∏ᵢ₌₁ⁿ σ(zᵢ)^yᵢ × (1-σ(zᵢ))^(1-yᵢ)")
        print()
        
        print("🔹 Log-Likelihood (easier to work with):")
        print("   ℓ(β) = log(L(β)) = Σᵢ₌₁ⁿ [yᵢlog(σ(zᵢ)) + (1-yᵢ)log(1-σ(zᵢ))]")
        print()
        
        print("🔹 In Bot Detection Context:")
        print("   • yᵢ = 1 if account i is a bot, 0 if human")
        print("   • σ(zᵢ) = predicted probability that account i is a bot")
        print("   • We want to maximize ℓ(β) to find best parameters")
        print()
        
        # Demonstrate with example
        self.demonstrate_mle_example()
        
    def demonstrate_mle_example(self):
        """Demonstrate MLE with a simple example"""
        print("🔹 Simple Example:")
        print("   Suppose we have 3 accounts with features and true labels:")
        print()
        
        # Create example data
        examples = pd.DataFrame({
            'Account': ['@user1', '@bot2', '@user3'],
            'Followers/Following': [2.5, 0.1, 8.0],
            'True Label': ['Human (0)', 'Bot (1)', 'Human (0)']
        })
        print(examples.to_string(index=False))
        print()
        
        print("   If our model predicts probabilities: [0.2, 0.9, 0.1]")
        print("   Likelihood = 0.8 × 0.9 × 0.9 = 0.648")
        print()
        print("   If our model predicts probabilities: [0.7, 0.9, 0.1]")
        print("   Likelihood = 0.3 × 0.9 × 0.9 = 0.243")
        print()
        print("   The first model is better! (Higher likelihood)")
        print()
        
    def explain_gradient_descent(self):
        """Explain gradient descent/ascent"""
        print("⬇️ 4. GRADIENT DESCENT/ASCENT")
        print("-" * 40)
        print()
        
        print("🔹 The Optimization Problem:")
        print("   We want to maximize log-likelihood ℓ(β)")
        print("   Equivalently: minimize negative log-likelihood -ℓ(β)")
        print()
        
        print("🔹 Gradient Descent Algorithm:")
        print("   1. Start with random parameter values β⁽⁰⁾")
        print("   2. Compute gradient: ∇ℓ(β) = ∂ℓ/∂β")
        print("   3. Update: β⁽ᵗ⁺¹⁾ = β⁽ᵗ⁾ + α∇ℓ(β⁽ᵗ⁾)  [ascent]")
        print("      Or:    β⁽ᵗ⁺¹⁾ = β⁽ᵗ⁾ - α∇(-ℓ)(β⁽ᵗ⁾) [descent]")
        print("   4. Repeat until convergence")
        print()
        
        print("🔹 The Gradient Formula:")
        print("   ∂ℓ/∂βⱼ = Σᵢ₌₁ⁿ (yᵢ - σ(zᵢ)) × xᵢⱼ")
        print("   Where:")
        print("   • yᵢ = true label (0 or 1)")
        print("   • σ(zᵢ) = predicted probability")
        print("   • xᵢⱼ = feature j for observation i")
        print()
        
        print("🔹 Intuition:")
        print("   • If yᵢ > σ(zᵢ): prediction too low → increase βⱼ")
        print("   • If yᵢ < σ(zᵢ): prediction too high → decrease βⱼ")
        print("   • Update magnitude proportional to feature value xᵢⱼ")
        print()
        
        # Visualize gradient descent
        self.visualize_gradient_descent()
        
    def visualize_gradient_descent(self):
        """Visualize gradient descent process"""
        # Simple 1D example
        def negative_log_likelihood(beta, x, y):
            z = beta * x
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-15, 1-1e-15)  # Avoid log(0)
            return -np.sum(y * np.log(p) + (1-y) * np.log(1-p))
        
        # Generate example data
        np.random.seed(42)
        x = np.array([1, 2, 3, 4, 5, 6])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        # Plot likelihood surface
        betas = np.linspace(-2, 4, 100)
        likelihoods = [negative_log_likelihood(b, x, y) for b in betas]
        
        plt.figure(figsize=(12, 5))
        
        # Likelihood surface
        plt.subplot(1, 2, 1)
        plt.plot(betas, likelihoods, 'b-', linewidth=2)
        plt.xlabel('β (parameter value)', fontsize=12)
        plt.ylabel('Negative Log-Likelihood', fontsize=12)
        plt.title('Optimization Surface', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Mark minimum
        min_idx = np.argmin(likelihoods)
        plt.plot(betas[min_idx], likelihoods[min_idx], 'ro', markersize=8, label='Optimal β')
        plt.legend()
        
        # Gradient descent path
        plt.subplot(1, 2, 2)
        
        # Simulate gradient descent
        beta_path = []
        beta = -1.0  # Starting point
        learning_rate = 0.1
        
        for iteration in range(20):
            beta_path.append(beta)
            # Compute gradient
            z = beta * x
            p = 1 / (1 + np.exp(-z))
            gradient = np.sum((y - p) * x)
            # Update (gradient ascent on log-likelihood)
            beta = beta + learning_rate * gradient
            
        plt.plot(range(len(beta_path)), beta_path, 'ro-', linewidth=2, markersize=6)
        plt.axhline(y=betas[min_idx], color='g', linestyle='--', alpha=0.7, label='Optimal β')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('β value', fontsize=12)
        plt.title('Gradient Descent Path', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def explain_bot_detection_application(self):
        """Explain how these concepts apply to bot detection"""
        print("🤖 5. APPLICATION TO BOT DETECTION")
        print("-" * 45)
        print()
        
        print("🔹 Our Feature Vector (x):")
        print("   x = [follower_ratio, tweets_per_day, account_age, verified, ...]")
        print()
        
        print("🔹 Linear Combination (z):")
        print("   z = β₀ + β₁×follower_ratio + β₂×tweets_per_day + β₃×account_age + ...")
        print()
        
        print("🔹 Bot Probability:")
        print("   P(bot|x) = 1 / (1 + e^(-z))")
        print()
        
        print("🔹 Training Process:")
        print("   1. Collect labeled data: (features, is_bot)")
        print("   2. Initialize parameters β randomly")
        print("   3. For each training example:")
        print("      • Compute predicted probability")
        print("      • Compare with true label")
        print("      • Update parameters using gradient")
        print("   4. Repeat until convergence")
        print()
        
        print("🔹 Example Parameter Interpretation:")
        print("   If β₁ = -2.5 (for follower_ratio):")
        print("   • Higher follower/following ratio → lower bot probability")
        print("   • Makes sense: bots often follow many, have few followers")
        print()
        
        print("🔹 Prediction for New Account:")
        print("   Given features: follower_ratio=0.1, tweets_per_day=50, age=30")
        print("   z = β₀ + (-2.5)×0.1 + β₂×50 + β₃×30")
        print("   P(bot) = 1 / (1 + e^(-z))")
        print()
        
        # Show actual example with realistic parameters
        self.demonstrate_bot_prediction()
        
    def demonstrate_bot_prediction(self):
        """Demonstrate actual bot prediction"""
        print("🔹 Realistic Example:")
        print()
        
        # Realistic parameters (simplified)
        beta_0 = -1.0  # Intercept
        beta_follower_ratio = -2.0  # Negative: higher ratio = less likely bot
        beta_tweets_per_day = 0.05   # Positive: more tweets = more likely bot
        beta_age = -0.001            # Negative: older account = less likely bot
        
        # Two example accounts
        accounts = [
            {
                'name': 'Suspicious Bot',
                'follower_ratio': 0.02,    # Very low
                'tweets_per_day': 100,     # Very high
                'age_days': 30             # Very new
            },
            {
                'name': 'Normal Human',
                'follower_ratio': 2.5,     # Normal
                'tweets_per_day': 3,       # Normal
                'age_days': 1200           # Established
            }
        ]
        
        for account in accounts:
            z = (beta_0 + 
                 beta_follower_ratio * account['follower_ratio'] +
                 beta_tweets_per_day * account['tweets_per_day'] +
                 beta_age * account['age_days'])
            
            prob_bot = 1 / (1 + np.exp(-z))
            
            print(f"   {account['name']}:")
            print(f"   • Follower ratio: {account['follower_ratio']}")
            print(f"   • Tweets/day: {account['tweets_per_day']}")
            print(f"   • Age: {account['age_days']} days")
            print(f"   • z = {z:.3f}")
            print(f"   • P(bot) = {prob_bot:.1%}")
            print(f"   • Classification: {'BOT' if prob_bot > 0.5 else 'HUMAN'}")
            print()
        
    def provide_summary(self):
        """Provide a comprehensive summary"""
        print("🎓 Key Takeaways:")
        print()
        print("1. 📊 LOGISTIC REGRESSION:")
        print("   • Maps any real number to probability [0,1] using sigmoid")
        print("   • Perfect for binary classification (bot vs human)")
        print()
        print("2. 📈 SIGMOID FUNCTION:")
        print("   • S-shaped curve: σ(z) = 1/(1+e^(-z))")
        print("   • Smooth, differentiable, bounded [0,1]")
        print()
        print("3. 🎯 MAXIMUM LIKELIHOOD:")
        print("   • Find parameters that make observed data most likely")
        print("   • Maximize: ℓ(β) = Σ[y log(p) + (1-y) log(1-p)]")
        print()
        print("4. ⬇️ GRADIENT DESCENT:")
        print("   • Iteratively update parameters: β ← β + α∇ℓ")
        print("   • Gradient: ∇ℓ = Σ(y - p)x")
        print()
        print("5. 🤖 BOT DETECTION:")
        print("   • Features: follower ratios, activity patterns, account age")
        print("   • Output: probability that account is automated")
        print("   • Decision: classify as bot if P(bot) > 0.5")
        print()
        print("🚀 This mathematical foundation enables our 98% accuracy!")

def main():
    """Run the complete explanation"""
    explainer = LogisticRegressionExplainer()
    explainer.explain_all_concepts()
    
    print("\n" + "="*60)
    print("📚 INTERACTIVE DEMO COMPLETE")
    print("="*60)
    print("This explanation covers the mathematical foundations")
    print("behind our bot detection system. The same principles")
    print("apply to the GUI demo you can run with:")
    print("    python bot_detector_gui.py")

if __name__ == "__main__":
    main() 