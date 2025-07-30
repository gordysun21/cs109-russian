"""
Quick Mathematical Concepts Demo for Bot Detection

A concise explanation of the key mathematical concepts for presentation purposes.
"""

import numpy as np
import matplotlib.pyplot as plt

def demo_sigmoid():
    """Quick sigmoid function demonstration"""
    print("🔹 THE SIGMOID FUNCTION")
    print("   Formula: P(bot) = 1 / (1 + e^(-z))")
    print("   Where z = β₀ + β₁×features₁ + β₂×features₂ + ...")
    print()
    
    # Quick visualization
    z = np.linspace(-6, 6, 100)
    p = 1 / (1 + np.exp(-z))
    
    plt.figure(figsize=(8, 5))
    plt.plot(z, p, 'b-', linewidth=3, label='Sigmoid Function')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Boundary')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('z (weighted sum of features)')
    plt.ylabel('P(bot)')
    plt.title('Sigmoid Function: Converting Features to Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    plt.text(-3, 0.2, 'Human\n(P < 0.5)', ha='center', fontsize=12, 
             bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7))
    plt.text(3, 0.8, 'Bot\n(P > 0.5)', ha='center', fontsize=12,
             bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7))
    
    plt.tight_layout()
    plt.show()

def demo_mle_concept():
    """Demonstrate MLE concept simply"""
    print("🔹 MAXIMUM LIKELIHOOD ESTIMATION (MLE)")
    print("   Goal: Find parameters that make our training data most likely")
    print()
    print("   Example: We have 3 accounts")
    print("   • Account A: Human (actual) → Model predicts 20% bot")
    print("   • Account B: Bot (actual)   → Model predicts 90% bot") 
    print("   • Account C: Human (actual) → Model predicts 15% bot")
    print()
    print("   Likelihood = (1-0.2) × 0.9 × (1-0.15) = 0.8 × 0.9 × 0.85 = 0.612")
    print()
    print("   MLE finds parameters that maximize this likelihood!")
    print()

def demo_gradient_descent():
    """Simple gradient descent explanation"""
    print("🔹 GRADIENT DESCENT")
    print("   How we find the best parameters:")
    print()
    print("   1. Start with random parameters")
    print("   2. Calculate error (how wrong our predictions are)")
    print("   3. Adjust parameters to reduce error")
    print("   4. Repeat until we find the best parameters")
    print()
    print("   It's like rolling a ball down a hill to find the bottom!")
    print()

def demo_bot_detection_math():
    """Show actual bot detection calculation"""
    print("🔹 ACTUAL BOT DETECTION CALCULATION")
    print()
    
    # Example parameters (simplified)
    print("   Learned Parameters:")
    print("   • β₀ (intercept) = -1.0")
    print("   • β₁ (follower_ratio) = -2.0")
    print("   • β₂ (tweets_per_day) = 0.05")
    print("   • β₃ (account_age) = -0.001")
    print()
    
    # Example account
    print("   Suspicious Account:")
    print("   • Follower/Following ratio = 0.1")
    print("   • Tweets per day = 80")
    print("   • Account age = 45 days")
    print()
    
    # Calculate
    z = -1.0 + (-2.0 * 0.1) + (0.05 * 80) + (-0.001 * 45)
    prob = 1 / (1 + np.exp(-z))
    
    print("   Calculation:")
    print(f"   z = -1.0 + (-2.0×0.1) + (0.05×80) + (-0.001×45)")
    print(f"   z = -1.0 - 0.2 + 4.0 - 0.045 = {z:.3f}")
    print(f"   P(bot) = 1/(1+e^(-{z:.3f})) = {prob:.1%}")
    print()
    print(f"   Result: {prob:.1%} chance this is a BOT! 🚨")
    print()

def main():
    """Run quick math demo"""
    print("🤖 MATHEMATICAL CONCEPTS IN BOT DETECTION")
    print("=" * 55)
    print()
    
    print("Our bot detector uses LOGISTIC REGRESSION")
    print("Here are the key mathematical concepts:")
    print()
    
    # 1. Sigmoid function
    demo_sigmoid()
    input("\nPress Enter to continue...")
    
    # 2. MLE
    demo_mle_concept()
    input("Press Enter to continue...")
    
    # 3. Gradient descent
    demo_gradient_descent()
    input("Press Enter to continue...")
    
    # 4. Actual calculation
    demo_bot_detection_math()
    
    print("🎯 KEY TAKEAWAY:")
    print("   Mathematics allows us to:")
    print("   • Convert any features into probabilities [0,1]")
    print("   • Learn from training data automatically")
    print("   • Make accurate predictions (98% accuracy!)")
    print()
    print("✨ This is the power of machine learning! ✨")

if __name__ == "__main__":
    main() 