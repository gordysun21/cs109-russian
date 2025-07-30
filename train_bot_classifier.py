"""
Train Bot Detection Classifier

This script trains a logistic regression model on the balanced bot detection dataset
to predict the probability that a tweet comes from a bot.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Data directory
BALANCED_DATA_DIR = Path("data/balanced_bot_detection")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('default')

def load_data():
    """Load the training, validation, and test datasets"""
    print("📥 Loading balanced bot detection datasets...")
    
    train_path = BALANCED_DATA_DIR / "train_set.csv"
    val_path = BALANCED_DATA_DIR / "validation_set.csv"
    test_path = BALANCED_DATA_DIR / "test_set.csv"
    
    if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
        raise FileNotFoundError("Dataset files not found. Run create_balanced_dataset.py first.")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"   Training set: {len(train_df):,} samples")
    print(f"   Validation set: {len(val_df):,} samples")
    print(f"   Test set: {len(test_df):,} samples")
    
    return train_df, val_df, test_df

def prepare_features(train_df, val_df, test_df):
    """Prepare features and target variables for modeling"""
    print("\n🔧 Preparing features for modeling...")
    
    # Define columns to exclude from features
    exclude_cols = ['bot_label', 'data_source', 'content', 'author', 'publish_date', 'language']
    
    # Get all potential feature columns
    all_cols = train_df.columns.tolist()
    feature_cols = [col for col in all_cols if col not in exclude_cols]
    
    print(f"   Initial feature columns: {feature_cols}")
    
    # Prepare training data
    X_train = train_df[feature_cols].copy()
    y_train = train_df['bot_label']
    
    # Prepare validation data
    X_val = val_df[feature_cols].copy()
    y_val = val_df['bot_label']
    
    # Prepare test data
    X_test = test_df[feature_cols].copy()
    y_test = test_df['bot_label']
    
    # Handle categorical variables
    categorical_cols = []
    numerical_cols = []
    
    for col in feature_cols:
        if X_train[col].dtype in ['object', 'category']:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    print(f"   Categorical features: {categorical_cols}")
    print(f"   Numerical features: {numerical_cols}")
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        print(f"   Encoding categorical feature: {col}")
        le = LabelEncoder()
        
        # Fit on training data
        X_train[col] = X_train[col].fillna('unknown')
        le.fit(X_train[col])
        
        # Transform all datasets
        X_train[col] = le.transform(X_train[col])
        
        # Handle new categories in val/test sets
        X_val[col] = X_val[col].fillna('unknown')
        X_val[col] = X_val[col].apply(lambda x: x if x in le.classes_ else 'unknown')
        X_val[col] = le.transform(X_val[col])
        
        X_test[col] = X_test[col].fillna('unknown')
        X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else 'unknown')
        X_test[col] = le.transform(X_test[col])
        
        label_encoders[col] = le
    
    # Handle missing values in numerical features
    for col in numerical_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_val[col] = X_val[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    
    # Convert boolean columns to integers
    for col in feature_cols:
        if X_train[col].dtype == 'bool':
            X_train[col] = X_train[col].astype(int)
            X_val[col] = X_val[col].astype(int)
            X_test[col] = X_test[col].astype(int)
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   ✅ Prepared {len(feature_cols)} features for modeling")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_cols, scaler, label_encoders

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train logistic regression model with hyperparameter tuning"""
    print("\n🤖 Training Logistic Regression model...")
    
    # Try different regularization strengths
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_score = 0
    best_model = None
    best_C = None
    
    for C in C_values:
        model = LogisticRegression(C=C, random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_score = model.score(X_val, y_val)
        print(f"   C={C}: Validation accuracy = {val_score:.4f}")
        
        if val_score > best_score:
            best_score = val_score
            best_model = model
            best_C = C
    
    print(f"   ✅ Best model: C={best_C}, Validation accuracy = {best_score:.4f}")
    return best_model

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest model for comparison"""
    print("\n🌲 Training Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    print(f"   ✅ Random Forest validation accuracy = {val_score:.4f}")
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation"""
    print(f"\n📊 Evaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of being a bot
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-score:  {f1:.4f}")
    print(f"   ROC AUC:   {auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def create_visualizations(lr_model, rf_model, X_test, y_test, lr_results, rf_results, feature_cols):
    """Create visualization plots for model evaluation"""
    print("\n📈 Creating evaluation visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Confusion Matrix - Logistic Regression
    cm_lr = confusion_matrix(y_test, lr_results['y_pred'])
    sns.heatmap(cm_lr, annot=True, fmt='d', ax=axes[0,0], cmap='Blues')
    axes[0,0].set_title('Logistic Regression\nConfusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    # 2. Confusion Matrix - Random Forest
    cm_rf = confusion_matrix(y_test, rf_results['y_pred'])
    sns.heatmap(cm_rf, annot=True, fmt='d', ax=axes[0,1], cmap='Greens')
    axes[0,1].set_title('Random Forest\nConfusion Matrix')
    axes[0,1].set_xlabel('Predicted')
    axes[0,1].set_ylabel('Actual')
    
    # 3. ROC Curves
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_results['y_pred_proba'])
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_results['y_pred_proba'])
    
    axes[0,2].plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_results["auc"]:.3f})')
    axes[0,2].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_results["auc"]:.3f})')
    axes[0,2].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    axes[0,2].set_xlabel('False Positive Rate')
    axes[0,2].set_ylabel('True Positive Rate')
    axes[0,2].set_title('ROC Curves')
    axes[0,2].legend()
    
    # 4. Feature Importance - Logistic Regression
    feature_importance_lr = abs(lr_model.coef_[0])
    top_features_idx = np.argsort(feature_importance_lr)[-10:]
    
    axes[1,0].barh(range(len(top_features_idx)), feature_importance_lr[top_features_idx])
    axes[1,0].set_yticks(range(len(top_features_idx)))
    axes[1,0].set_yticklabels([feature_cols[i] for i in top_features_idx])
    axes[1,0].set_title('Top 10 Features\n(Logistic Regression)')
    axes[1,0].set_xlabel('Coefficient Magnitude')
    
    # 5. Feature Importance - Random Forest
    feature_importance_rf = rf_model.feature_importances_
    top_features_idx_rf = np.argsort(feature_importance_rf)[-10:]
    
    axes[1,1].barh(range(len(top_features_idx_rf)), feature_importance_rf[top_features_idx_rf])
    axes[1,1].set_yticks(range(len(top_features_idx_rf)))
    axes[1,1].set_yticklabels([feature_cols[i] for i in top_features_idx_rf])
    axes[1,1].set_title('Top 10 Features\n(Random Forest)')
    axes[1,1].set_xlabel('Feature Importance')
    
    # 6. Bot Probability Distribution
    human_probs = lr_results['y_pred_proba'][y_test == 0]
    bot_probs = lr_results['y_pred_proba'][y_test == 1]
    
    axes[1,2].hist(human_probs, bins=30, alpha=0.7, label='Human tweets', color='skyblue')
    axes[1,2].hist(bot_probs, bins=30, alpha=0.7, label='Bot tweets', color='lightcoral')
    axes[1,2].set_xlabel('Predicted Bot Probability')
    axes[1,2].set_ylabel('Count')
    axes[1,2].set_title('Bot Probability Distribution')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig('bot_detection_model_evaluation.png', dpi=300, bbox_inches='tight')
    print("   📊 Saved evaluation plots to 'bot_detection_model_evaluation.png'")

def predict_bot_probability(model, scaler, feature_cols, label_encoders, sample_data):
    """
    Predict bot probability for new data
    
    Args:
        model: Trained classifier
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
        label_encoders: Dictionary of LabelEncoders for categorical features
        sample_data: Dictionary with feature values
    
    Returns:
        float: Probability of being a bot (0-1)
    """
    # Create dataframe from sample data
    df_sample = pd.DataFrame([sample_data])
    
    # Ensure all features are present
    for col in feature_cols:
        if col not in df_sample.columns:
            df_sample[col] = 0  # Default value
    
    # Apply label encoding for categorical features
    for col, le in label_encoders.items():
        if col in df_sample.columns:
            val = df_sample[col].iloc[0]
            if val in le.classes_:
                df_sample[col] = le.transform([val])[0]
            else:
                df_sample[col] = le.transform(['unknown'])[0]
    
    # Select and scale features
    X_sample = df_sample[feature_cols].fillna(0)
    X_sample_scaled = scaler.transform(X_sample)
    
    # Predict probability
    bot_probability = model.predict_proba(X_sample_scaled)[0, 1]
    
    return bot_probability

def save_models(lr_model, rf_model, scaler, feature_cols, label_encoders):
    """Save trained models and preprocessing objects"""
    print("\n💾 Saving trained models...")
    
    joblib.dump(lr_model, MODELS_DIR / "logistic_regression_bot_detector.pkl")
    joblib.dump(rf_model, MODELS_DIR / "random_forest_bot_detector.pkl")
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.pkl")
    joblib.dump(label_encoders, MODELS_DIR / "label_encoders.pkl")
    
    # Save feature names
    with open(MODELS_DIR / "feature_columns.txt", 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    
    print(f"   ✅ Models saved to {MODELS_DIR}/")

def demonstrate_predictions(lr_model, scaler, feature_cols, label_encoders):
    """Demonstrate bot probability predictions on sample data"""
    print("\n🎯 DEMONSTRATING BOT PROBABILITY PREDICTIONS")
    print("=" * 50)
    
    # Sample bot-like account
    bot_sample = {
        'followers': 100,
        'following': 5000,
        'updates': 10000,
        'retweet': 1,
        'account_type': 'Right',
        'region': 'Unknown'
    }
    
    # Sample human-like account
    human_sample = {
        'followers': 500,
        'following': 300,
        'updates': 1200,
        'retweet': 0,
        'account_type': 'human',
        'region': 'United States'
    }
    
    bot_prob = predict_bot_probability(lr_model, scaler, feature_cols, label_encoders, bot_sample)
    human_prob = predict_bot_probability(lr_model, scaler, feature_cols, label_encoders, human_sample)
    
    print(f"Bot-like account probability:   {bot_prob:.1%} chance of being a bot")
    print(f"Human-like account probability: {human_prob:.1%} chance of being a bot")

def main():
    """Main function to train and evaluate bot detection models"""
    print("🤖 BOT DETECTION CLASSIFIER TRAINER")
    print("=" * 50)
    
    try:
        # Load data
        train_df, val_df, test_df = load_data()
        
        # Prepare features
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler, label_encoders = prepare_features(
            train_df, val_df, test_df
        )
        
        # Train models
        lr_model = train_logistic_regression(X_train, y_train, X_val, y_val)
        rf_model = train_random_forest(X_train, y_train, X_val, y_val)
        
        # Evaluate models
        lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        
        # Create visualizations
        create_visualizations(lr_model, rf_model, X_test, y_test, lr_results, rf_results, feature_cols)
        
        # Save models
        save_models(lr_model, rf_model, scaler, feature_cols, label_encoders)
        
        # Demonstrate predictions
        demonstrate_predictions(lr_model, scaler, feature_cols, label_encoders)
        
        print(f"\n✅ SUCCESS! Bot detection models trained and evaluated")
        print(f"\n🎯 KEY RESULTS:")
        print(f"   Logistic Regression: {lr_results['accuracy']:.1%} accuracy, {lr_results['auc']:.3f} AUC")
        print(f"   Random Forest:       {rf_results['accuracy']:.1%} accuracy, {rf_results['auc']:.3f} AUC")
        print(f"\n📁 Files created:")
        print(f"   • {MODELS_DIR}/logistic_regression_bot_detector.pkl")
        print(f"   • {MODELS_DIR}/random_forest_bot_detector.pkl")
        print(f"   • {MODELS_DIR}/feature_scaler.pkl")
        print(f"   • {MODELS_DIR}/label_encoders.pkl")
        print(f"   • bot_detection_model_evaluation.png")
        print(f"\n🎉 You can now predict bot probabilities for any tweet!")
        
    except Exception as e:
        print(f"\n❌ Error training models: {e}")
        print("\nMake sure you have:")
        print("1. Run download_legitimate_tweets.py")
        print("2. Run create_balanced_dataset.py")
        print("3. Check that the data files exist")

if __name__ == "__main__":
    main() 