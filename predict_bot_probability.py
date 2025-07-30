"""
Bot Probability Predictor

This script loads the trained bot detection model and allows you to 
predict the probability that any tweet or account is from a bot.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Model directory
MODELS_DIR = Path("models")

def load_trained_models():
    """Load the trained models and preprocessing objects"""
    print("🔧 Loading trained bot detection models...")
    
    try:
        lr_model = joblib.load(MODELS_DIR / "logistic_regression_bot_detector.pkl")
        rf_model = joblib.load(MODELS_DIR / "random_forest_bot_detector.pkl")
        scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")
        label_encoders = joblib.load(MODELS_DIR / "label_encoders.pkl")
        
        with open(MODELS_DIR / "feature_columns.txt", 'r') as f:
            feature_cols = [line.strip() for line in f.readlines()]
        
        print("   ✅ Models loaded successfully!")
        return lr_model, rf_model, scaler, label_encoders, feature_cols
        
    except FileNotFoundError as e:
        print(f"   ❌ Model files not found: {e}")
        print("   Please run train_bot_classifier.py first to train the models.")
        return None, None, None, None, None

def predict_bot_probability(model, scaler, feature_cols, label_encoders, account_data, model_name="Logistic Regression"):
    """
    Predict bot probability for account data
    
    Args:
        model: Trained classifier
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
        label_encoders: Dictionary of LabelEncoders
        account_data: Dictionary with account features
        model_name: Name of the model for display
    
    Returns:
        float: Probability of being a bot (0-1)
    """
    # Create dataframe from account data
    df_sample = pd.DataFrame([account_data])
    
    # Ensure all features are present with defaults
    feature_defaults = {
        'followers': 100,
        'following': 100,
        'updates': 500,
        'retweet': 0,
        'account_type': 'human',
        'region': 'United States',
        'follower_following_ratio': 1.0,
        'content_length': 100,
        'has_url': 0,
        'hashtag_count': 1,
        'mention_count': 0,
        'exclamation_count': 0,
        'question_count': 0,
        'account_age_days': 365,
        'tweets_per_day': 2
    }
    
    for col in feature_cols:
        if col not in df_sample.columns:
            df_sample[col] = feature_defaults.get(col, 0)
    
    # Apply label encoding for categorical features
    for col, le in label_encoders.items():
        if col in df_sample.columns:
            val = str(df_sample[col].iloc[0])
            if val in le.classes_:
                df_sample[col] = le.transform([val])[0]
            else:
                df_sample[col] = le.transform(['unknown'])[0]
    
    # Select features in correct order
    X_sample = df_sample[feature_cols].fillna(0)
    
    # Scale features
    X_sample_scaled = scaler.transform(X_sample)
    
    # Predict probability
    bot_probability = model.predict_proba(X_sample_scaled)[0, 1]
    
    return bot_probability

def interactive_prediction():
    """Interactive interface for bot probability prediction"""
    print("\n🤖 INTERACTIVE BOT PROBABILITY PREDICTOR")
    print("=" * 50)
    
    # Load models
    lr_model, rf_model, scaler, label_encoders, feature_cols = load_trained_models()
    
    if lr_model is None:
        return
    
    print("\nYou can enter account information to predict bot probability.")
    print("Press Enter to use default values for any field.")
    print("Type 'quit' to exit.\n")
    
    while True:
        print("-" * 40)
        print("Enter account information:")
        
        # Get user input
        try:
            followers = input("Followers count (default: 500): ").strip()
            followers = int(followers) if followers else 500
            
            following = input("Following count (default: 300): ").strip()
            following = int(following) if following else 300
            
            updates = input("Total tweets/updates (default: 1200): ").strip()
            updates = int(updates) if updates else 1200
            
            content_length = input("Average tweet length (default: 100): ").strip()
            content_length = int(content_length) if content_length else 100
            
            has_url = input("Contains URLs? (y/n, default: n): ").strip().lower()
            has_url = 1 if has_url == 'y' else 0
            
            hashtags = input("Average hashtags per tweet (default: 1): ").strip()
            hashtags = int(hashtags) if hashtags else 1
            
            mentions = input("Average mentions per tweet (default: 0): ").strip()
            mentions = int(mentions) if mentions else 0
            
            retweet = input("Is this a retweet? (y/n, default: n): ").strip().lower()
            retweet = 1 if retweet == 'y' else 0
            
            account_type = input("Account type (human/Right/Left/Commercial, default: human): ").strip()
            account_type = account_type if account_type else 'human'
            
            region = input("Region (default: United States): ").strip()
            region = region if region else 'United States'
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except ValueError:
            print("Invalid input. Please enter numbers for numerical fields.")
            continue
        
        # Create account data
        account_data = {
            'followers': followers,
            'following': following,
            'updates': updates,
            'retweet': retweet,
            'account_type': account_type,
            'region': region,
            'follower_following_ratio': followers / (following + 1),
            'content_length': content_length,
            'has_url': has_url,
            'hashtag_count': hashtags,
            'mention_count': mentions,
            'exclamation_count': 0,
            'question_count': 0,
            'account_age_days': 365,
            'tweets_per_day': updates / 365
        }
        
        # Predict with both models
        lr_prob = predict_bot_probability(lr_model, scaler, feature_cols, label_encoders, account_data, "Logistic Regression")
        rf_prob = predict_bot_probability(rf_model, scaler, feature_cols, label_encoders, account_data, "Random Forest")
        
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"   Logistic Regression: {lr_prob:.1%} chance of being a bot")
        print(f"   Random Forest:       {rf_prob:.1%} chance of being a bot")
        print(f"   Average prediction:  {(lr_prob + rf_prob)/2:.1%} chance of being a bot")
        
        # Interpretation
        avg_prob = (lr_prob + rf_prob) / 2
        if avg_prob < 0.1:
            print("   🟢 Very likely HUMAN")
        elif avg_prob < 0.3:
            print("   🟡 Probably human")
        elif avg_prob < 0.7:
            print("   🟠 Uncertain - could be either")
        elif avg_prob < 0.9:
            print("   🔴 Probably bot")
        else:
            print("   🚨 Very likely BOT")
        
        # Continue?
        continue_pred = input("\nPredict another account? (y/n): ").strip().lower()
        if continue_pred != 'y':
            break
    
    print("\nThanks for using the bot detection system!")

def batch_prediction_examples():
    """Run predictions on several example accounts"""
    print("\n🧪 BATCH PREDICTION EXAMPLES")
    print("=" * 40)
    
    # Load models
    lr_model, rf_model, scaler, label_encoders, feature_cols = load_trained_models()
    
    if lr_model is None:
        return
    
    # Example accounts
    examples = [
        {
            'name': 'Typical Human User',
            'data': {
                'followers': 450,
                'following': 320,
                'updates': 1800,
                'content_length': 95,
                'has_url': 0,
                'hashtag_count': 1,
                'mention_count': 0,
                'retweet': 0,
                'account_type': 'human',
                'region': 'United States'
            }
        },
        {
            'name': 'Suspicious Bot-like Account',
            'data': {
                'followers': 89,
                'following': 4500,
                'updates': 12000,
                'content_length': 140,
                'has_url': 1,
                'hashtag_count': 4,
                'mention_count': 3,
                'retweet': 1,
                'account_type': 'Right',
                'region': 'Unknown'
            }
        },
        {
            'name': 'Popular Influencer',
            'data': {
                'followers': 15000,
                'following': 800,
                'updates': 3200,
                'content_length': 120,
                'has_url': 1,
                'hashtag_count': 2,
                'mention_count': 1,
                'retweet': 0,
                'account_type': 'human',
                'region': 'United States'
            }
        },
        {
            'name': 'Commercial Account',
            'data': {
                'followers': 2500,
                'following': 150,
                'updates': 5500,
                'content_length': 110,
                'has_url': 1,
                'hashtag_count': 3,
                'mention_count': 0,
                'retweet': 0,
                'account_type': 'Commercial',
                'region': 'United States'
            }
        }
    ]
    
    for example in examples:
        print(f"\n📊 {example['name']}:")
        
        # Add calculated features
        data = example['data'].copy()
        data['follower_following_ratio'] = data['followers'] / (data['following'] + 1)
        data['tweets_per_day'] = data['updates'] / 365
        data['account_age_days'] = 365
        data['exclamation_count'] = 0
        data['question_count'] = 0
        
        # Predict
        lr_prob = predict_bot_probability(lr_model, scaler, feature_cols, label_encoders, data)
        rf_prob = predict_bot_probability(rf_model, scaler, feature_cols, label_encoders, data)
        
        print(f"   Logistic Regression: {lr_prob:.1%} bot probability")
        print(f"   Random Forest:       {rf_prob:.1%} bot probability")
        print(f"   Average:             {(lr_prob + rf_prob)/2:.1%} bot probability")

def main():
    """Main function"""
    print("🤖 BOT PROBABILITY PREDICTOR")
    print("=" * 40)
    print("This tool uses machine learning to predict the probability")
    print("that a Twitter account is a bot based on account features.")
    
    while True:
        print("\nChoose an option:")
        print("1. Interactive prediction (enter account data)")
        print("2. Run example predictions")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            interactive_prediction()
        elif choice == '2':
            batch_prediction_examples()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 