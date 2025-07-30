"""
Interactive Bot Detector

A simple program that allows you to enter account information 
and get bot probability predictions using our trained models.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import os

# Setup paths
MODELS_DIR = Path("models")

def load_trained_models():
    """Load the trained models and preprocessing objects"""
    try:
        print("🤖 Loading trained bot detection models...")
        
        # Load models
        lr_model = joblib.load(MODELS_DIR / "logistic_regression_bot_detector.pkl")
        rf_model = joblib.load(MODELS_DIR / "random_forest_bot_detector.pkl")
        scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")
        label_encoders = joblib.load(MODELS_DIR / "label_encoders.pkl")
        
        # Load feature names
        with open(MODELS_DIR / "feature_columns.txt", 'r') as f:
            feature_cols = [line.strip() for line in f.readlines()]
        
        print("   ✅ Models loaded successfully!")
        return lr_model, rf_model, scaler, label_encoders, feature_cols
        
    except FileNotFoundError as e:
        print(f"❌ Error: Model files not found.")
        print(f"Please run 'python train_bot_classifier.py' first to train the models.")
        print(f"Missing file: {e}")
        return None, None, None, None, None

def get_user_input():
    """Get account information from user input with helpful prompts"""
    print("\n" + "="*60)
    print("🔍 ENTER ACCOUNT INFORMATION FOR BOT DETECTION")
    print("="*60)
    print("Enter the account details below. Press Enter for default values.")
    print()
    
    account_data = {}
    
    # Basic account metrics
    print("📊 BASIC ACCOUNT METRICS:")
    try:
        followers = input("   Followers count (default: 500): ").strip()
        account_data['followers'] = int(followers) if followers else 500
        
        following = input("   Following count (default: 300): ").strip()
        account_data['following'] = int(following) if following else 300
        
        updates = input("   Total tweets/updates (default: 1200): ").strip()
        account_data['updates'] = int(updates) if updates else 1200
        
    except ValueError:
        print("⚠️  Invalid number entered. Using default values.")
        account_data.update({'followers': 500, 'following': 300, 'updates': 1200})
    
    # Tweet content metrics
    print("\n📝 TWEET CONTENT METRICS:")
    try:
        word_count = input("   Average words per tweet (default: 15): ").strip()
        account_data['word_count'] = float(word_count) if word_count else 15.0
        
        hashtag_count = input("   Average hashtags per tweet (default: 1): ").strip()
        account_data['hashtag_count'] = float(hashtag_count) if hashtag_count else 1.0
        
        mention_count = input("   Average mentions per tweet (default: 0.5): ").strip()
        account_data['mention_count'] = float(mention_count) if mention_count else 0.5
        
        url_count = input("   Average URLs per tweet (default: 0.2): ").strip()
        account_data['url_count'] = float(url_count) if url_count else 0.2
        
    except ValueError:
        print("⚠️  Invalid number entered. Using default values.")
        account_data.update({
            'word_count': 15.0, 'hashtag_count': 1.0, 
            'mention_count': 0.5, 'url_count': 0.2
        })
    
    # Boolean features
    print("\n✅ ACCOUNT FEATURES:")
    retweet = input("   Does account frequently retweet? (y/n, default: n): ").strip().lower()
    account_data['retweet'] = 1 if retweet.startswith('y') else 0
    
    has_url = input("   Do tweets often contain URLs? (y/n, default: n): ").strip().lower()
    account_data['has_url'] = 1 if has_url.startswith('y') else 0
    
    verified = input("   Is account verified? (y/n, default: n): ").strip().lower()
    account_data['verified'] = 1 if verified.startswith('y') else 0
    
    # Categorical features
    print("\n📍 ACCOUNT CLASSIFICATION:")
    print("   Account types: Right, Left, Fearmonger, HashtagGamer, LeftTroll, RightTroll, Unknown")
    account_type = input("   Account type (default: Unknown): ").strip()
    account_data['account_type'] = account_type if account_type else 'Unknown'
    
    print("   Regions: United States, Unknown, etc.")
    region = input("   Account region (default: Unknown): ").strip()
    account_data['region'] = region if region else 'Unknown'
    
    return account_data

def process_account_data(account_data, feature_cols, label_encoders):
    """Process the account data into features for prediction"""
    
    # Create a copy to avoid modifying original
    processed_data = account_data.copy()
    
    # Calculate engineered features
    processed_data['follower_following_ratio'] = account_data['followers'] / (account_data['following'] + 1)
    processed_data['following_follower_ratio'] = account_data['following'] / (account_data['followers'] + 1)
    
    # Estimate account age (assume newer accounts tweet more frequently)
    # For demo purposes, estimate based on tweet frequency
    estimated_days = max(1, account_data['updates'] / 5)  # Assume ~5 tweets per day average
    processed_data['tweets_per_day'] = account_data['updates'] / estimated_days
    
    # Create DataFrame
    df = pd.DataFrame([processed_data])
    
    # Ensure all required features are present
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Apply label encoding for categorical features
    for col, le in label_encoders.items():
        if col in df.columns:
            val = df[col].iloc[0]
            if val in le.classes_:
                df[col] = le.transform([val])[0]
            else:
                # Handle unknown categories
                df[col] = le.transform(['Unknown'])[0] if 'Unknown' in le.classes_ else 0
    
    # Select features in correct order and fill missing values
    X = df[feature_cols].fillna(0)
    
    return X

def predict_bot_probability(models_data, account_data):
    """Make bot predictions using both models"""
    lr_model, rf_model, scaler, label_encoders, feature_cols = models_data
    
    # Process the input data
    X = process_account_data(account_data, feature_cols, label_encoders)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    lr_prob = lr_model.predict_proba(X_scaled)[0, 1]
    rf_prob = rf_model.predict_proba(X_scaled)[0, 1]
    
    # Get binary predictions
    lr_pred = lr_model.predict(X_scaled)[0]
    rf_pred = rf_model.predict(X_scaled)[0]
    
    return {
        'logistic_regression': {
            'probability': lr_prob,
            'prediction': 'BOT' if lr_pred == 1 else 'HUMAN',
            'confidence': max(lr_prob, 1 - lr_prob)
        },
        'random_forest': {
            'probability': rf_prob,
            'prediction': 'BOT' if rf_pred == 1 else 'HUMAN',
            'confidence': max(rf_prob, 1 - rf_prob)
        }
    }

def display_results(account_data, predictions):
    """Display the prediction results in a user-friendly format"""
    print("\n" + "="*60)
    print("🎯 BOT DETECTION RESULTS")
    print("="*60)
    
    # Show input summary
    print(f"\n📊 ACCOUNT SUMMARY:")
    print(f"   • Followers: {account_data['followers']:,}")
    print(f"   • Following: {account_data['following']:,}")
    print(f"   • Total tweets: {account_data['updates']:,}")
    print(f"   • Follower/Following ratio: {account_data['followers']/(account_data['following']+1):.2f}")
    print(f"   • Account type: {account_data['account_type']}")
    print(f"   • Has URLs: {'Yes' if account_data['has_url'] else 'No'}")
    print(f"   • Verified: {'Yes' if account_data['verified'] else 'No'}")
    
    # Show predictions
    lr_result = predictions['logistic_regression']
    rf_result = predictions['random_forest']
    
    print(f"\n🤖 PREDICTION RESULTS:")
    print(f"   📈 Logistic Regression:")
    print(f"      • Bot probability: {lr_result['probability']:.1%}")
    print(f"      • Classification: {lr_result['prediction']}")
    print(f"      • Confidence: {lr_result['confidence']:.1%}")
    
    print(f"\n   🌲 Random Forest:")
    print(f"      • Bot probability: {rf_result['probability']:.1%}")
    print(f"      • Classification: {rf_result['prediction']}")
    print(f"      • Confidence: {rf_result['confidence']:.1%}")
    
    # Overall assessment
    avg_prob = (lr_result['probability'] + rf_result['probability']) / 2
    consensus = "BOT" if avg_prob > 0.5 else "HUMAN"
    
    print(f"\n🎯 CONSENSUS PREDICTION:")
    print(f"   • Average bot probability: {avg_prob:.1%}")
    print(f"   • Final classification: {consensus}")
    
    if avg_prob > 0.8:
        print("   • 🚨 HIGH BOT PROBABILITY - Likely automated account")
    elif avg_prob > 0.6:
        print("   • ⚠️  MODERATE BOT PROBABILITY - Suspicious activity")
    elif avg_prob > 0.4:
        print("   • 🤔 UNCERTAIN - Mixed signals")
    elif avg_prob > 0.2:
        print("   • ✅ LOW BOT PROBABILITY - Likely human account")
    else:
        print("   • ✅ VERY LOW BOT PROBABILITY - Almost certainly human")

def show_examples():
    """Show some example account types for reference"""
    print("\n📋 EXAMPLE ACCOUNT TYPES:")
    print("-" * 40)
    
    examples = {
        "Typical Bot Account": {
            "followers": 50, "following": 2000, "updates": 5000,
            "word_count": 8, "hashtag_count": 5, "has_url": "Yes",
            "retweet": "Yes", "account_type": "Right"
        },
        "Normal Human Account": {
            "followers": 300, "following": 150, "updates": 800,
            "word_count": 20, "hashtag_count": 1, "has_url": "Sometimes",
            "retweet": "Sometimes", "account_type": "Unknown"
        },
        "Influencer Account": {
            "followers": 10000, "following": 500, "updates": 2000,
            "word_count": 25, "hashtag_count": 3, "has_url": "Yes",
            "retweet": "No", "account_type": "Unknown"
        }
    }
    
    for account_type, details in examples.items():
        print(f"\n{account_type}:")
        for key, value in details.items():
            print(f"   • {key}: {value}")

def main():
    """Main interactive bot detection program"""
    print("🤖 INTERACTIVE BOT DETECTOR")
    print("=" * 40)
    print("This program analyzes account characteristics to detect potential bots.")
    print()
    
    # Load models
    models_data = load_trained_models()
    if models_data[0] is None:
        return
    
    while True:
        print("\nOptions:")
        print("1. Analyze an account")
        print("2. Show example accounts")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            # Get account information
            account_data = get_user_input()
            
            # Make predictions
            try:
                predictions = predict_bot_probability(models_data, account_data)
                
                # Display results
                display_results(account_data, predictions)
                
            except Exception as e:
                print(f"❌ Error making prediction: {e}")
                print("Please check your input values and try again.")
        
        elif choice == '2':
            show_examples()
        
        elif choice == '3':
            print("\n👋 Thank you for using the Interactive Bot Detector!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
        
        # Ask if user wants to continue
        if choice in ['1', '2']:
            continue_choice = input("\nWould you like to analyze another account? (y/n): ").strip().lower()
            if not continue_choice.startswith('y'):
                print("\n👋 Thank you for using the Interactive Bot Detector!")
                break

if __name__ == "__main__":
    main() 