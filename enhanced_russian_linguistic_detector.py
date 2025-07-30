"""
Enhanced Russian Linguistic Bot Detector with Bootstrapping

This enhanced version creates a more sensitive detector that flags tweets as Russian 
if they contain even 1-2 linguistic infractions. Uses bootstrapping to ensure robust results.
"""

import pandas as pd
import numpy as np
import re
import string
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib
from data_utils import load_single_file, get_data_files
import warnings
warnings.filterwarnings('ignore')

# Data directories
BOT_DATA_DIR = Path("data/russian_troll_tweets")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class EnhancedRussianLinguisticExtractor:
    """
    Enhanced extractor with more sensitive pattern detection
    """
    
    def __init__(self):
        # More comprehensive Russian linguistic patterns
        self.critical_patterns = {
            # High-confidence Russian indicators (1 occurrence = likely Russian)
            'definitive_russian_patterns': [
                r'\b(actual|actually)\b(?=.*\b(topic|problem|situation|issue)\b)',  # "actual topic"
                r'\b(make)\s+(sport|sports)\b',                                     # "make sport"
                r'\b(depends|rely|count)\s+(from|of)\b',                           # "depends from"
                r'\b(different|separate)\s+to\b',                                  # "different to"
                r'\b(participate)\s+to\b',                                         # "participate to"
                r'\b(take)\s+(decision|conclusion)\b',                             # "take decision"
                r'\b(put)\s+(question|problem)\b',                                 # "put question"
                r'\b(very)\s+(actual|genius|sympathy)\b',                          # False friends
                r'\bwill\s+be\s+to\s+\w+\b',                                      # "will be to do"
                r'\b(am|is|are)\s+(knowing|understanding|believing|thinking)\b',   # Progressive statives
                r'\bmake\s+(homework|study)\b',                                    # "make homework"
            ],
            
            # Moderate confidence patterns (2-3 occurrences = likely Russian)
            'moderate_russian_patterns': [
                r'\b(very|quite|really)\s+(good|bad|nice|terrible)\s+(thing|person|idea|situation)\b',
                r'\b(all|many|some)\s+this\b',                                     # "all this"
                r'\b(what|which)\s+(kind|type)\s+of\b',                            # Word order
                r'\b(internet|web)\b',                                             # Lowercase internet
                r'\b(russian|american|ukrainian|german|french)\b',                 # Lowercase nationalities
                r'\b(consist)\s+from\b',                                           # "consist from"
                r'\b(afraid)\s+from\b',                                            # "afraid from"
                r'\b(have|has)\s+(wrote|came|went|done)\b',                        # Perfect tense errors
                r'\bin\s+(morning|evening|afternoon)\b',                           # "in morning"
                r'\bgo\s+to\s+(school|university|work)\b',                         # Missing articles
            ],
            
            # Weak indicators (need multiple occurrences)
            'weak_russian_patterns': [
                r'[.!?]\s+[a-z]',                                                  # Lowercase after punctuation
                r'\s+,\s+',                                                        # Spaces before commas
                r'[.!?]{2,}',                                                      # Multiple punctuation
                r'\b(definately|recieve|seperate|occured|wether|wich|teh|thier)\b', # Common misspellings
                r'\b(loose)\b(?=.*\b(game|match|competition)\b)',                  # "loose" instead of "lose"
                r'\b(its)\b(?=.*\b(important|necessary|possible)\b)',              # "its" without apostrophe
            ]
        }
        
        # Generate synthetic Russian-influenced text
        self.synthetic_patterns = [
            "This situation is very actual for our country today.",
            "I will be to make sport in morning.",
            "American people must understand that democracy depends from people.",
            "We need to take decision about this actual problem.",
            "Different to other countries, we have different situation.",
            "In internet you can find many informations about this topic.",
            "russian people are very good at making sport activities.",
            "I am knowing that this is important question for government.",
            "What kind of problems do american politicians have with this?",
            "Very actual topic in modern world is economic crisis.",
            "People must make homework before they participate to elections.",
            "This film is quite good thing for understanding american culture.",
            "Many informations can be found in internet about political situation.",
            "We must put question: what is actual problem in Ukraine today?",
            "Different from other countries, Russia has own economic system."
        ]

    def extract_features(self, text):
        """Extract enhanced linguistic features with sensitivity weighting"""
        if pd.isna(text) or not isinstance(text, str):
            text = ""
        
        text_lower = text.lower().strip()
        features = {}
        
        # Basic text statistics
        words = text_lower.split()
        features['text_length'] = len(text)
        features['word_count'] = len(words)
        features['sentence_count'] = max(1, len(re.findall(r'[.!?]+', text)))
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Critical Russian patterns (high weight)
        critical_count = 0
        for pattern in self.critical_patterns['definitive_russian_patterns']:
            matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
            critical_count += matches
        
        features['critical_russian_patterns'] = critical_count
        features['critical_pattern_density'] = critical_count / max(1, features['word_count'])
        
        # Moderate Russian patterns
        moderate_count = 0
        for pattern in self.critical_patterns['moderate_russian_patterns']:
            matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
            moderate_count += matches
        
        features['moderate_russian_patterns'] = moderate_count
        features['moderate_pattern_density'] = moderate_count / max(1, features['word_count'])
        
        # Weak Russian patterns
        weak_count = 0
        for pattern in self.critical_patterns['weak_russian_patterns']:
            matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
            weak_count += matches
        
        features['weak_russian_patterns'] = weak_count
        features['weak_pattern_density'] = weak_count / max(1, features['word_count'])
        
        # Combined pattern score (weighted)
        features['weighted_russian_score'] = (
            critical_count * 3.0 +      # Critical patterns heavily weighted
            moderate_count * 1.5 +       # Moderate patterns moderately weighted  
            weak_count * 0.5            # Weak patterns lightly weighted
        )
        
        # Normalized weighted score
        features['normalized_russian_score'] = features['weighted_russian_score'] / max(1, features['word_count'])
        
        # Simple rule-based classification
        features['rule_based_russian'] = 1 if (critical_count >= 1 or moderate_count >= 2 or weak_count >= 3) else 0
        
        # Article analysis (Russian speakers struggle with articles)
        articles = len(re.findall(r'\b(a|an|the)\b', text_lower))
        features['article_density'] = articles / max(1, features['word_count'])
        features['missing_articles'] = len(re.findall(r'\b(went\s+to|at|in|on)\s+[aeiou]', text_lower))
        
        # Grammar complexity
        features['complex_sentences'] = len(re.findall(r'\b(that|which|who|when|where|because|although)\b', text_lower))
        features['comma_usage'] = text.count(',') / max(1, features['sentence_count'])
        
        # Repetition patterns
        if len(words) > 1:
            word_freq = Counter(words)
            features['word_repetition'] = (len(words) - len(set(words))) / len(words)
            features['max_word_freq'] = max(word_freq.values()) / len(words)
        else:
            features['word_repetition'] = 0
            features['max_word_freq'] = 0
        
        # Punctuation patterns
        features['exclamation_density'] = text.count('!') / max(1, features['text_length'])
        features['question_density'] = text.count('?') / max(1, features['text_length'])
        features['capitalization_errors'] = len(re.findall(r'[.!?]\s+[a-z]', text))
        
        return features

    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic Russian-influenced text samples"""
        synthetic_samples = []
        
        # Generate samples with varying levels of Russian influence
        for i in range(n_samples):
            if i < n_samples // 3:
                # Heavy Russian influence
                base_text = np.random.choice(self.synthetic_patterns, 2)
                text = ' '.join(base_text)
                label = 1
            elif i < 2 * n_samples // 3:
                # Moderate Russian influence  
                base_text = np.random.choice(self.synthetic_patterns, 1)
                normal_text = f"This is a normal sentence about the topic. {base_text[0]} Thank you for reading."
                text = normal_text
                label = 1
            else:
                # Normal English (control group)
                normal_texts = [
                    "I'm really excited about the new movie coming out next week.",
                    "The weather is beautiful today and I'm planning to go for a walk.",
                    "I love reading books and watching documentaries about history.",
                    "The presentation went really well and everyone seemed engaged.",
                    "I'm looking forward to the weekend and spending time with family.",
                    "The restaurant has excellent food and great customer service.",
                    "I need to finish my homework before going out with friends tonight.",
                    "The concert was amazing and the musicians were incredibly talented.",
                ]
                text = np.random.choice(normal_texts)
                label = 0
            
            synthetic_samples.append({'text': text, 'label': label, 'source': 'synthetic'})
        
        return pd.DataFrame(synthetic_samples)

def load_and_prepare_data(max_samples=50000):
    """Load real Russian bot data and add synthetic samples"""
    print("📥 Loading Russian bot tweets and generating synthetic data...")
    
    # Load real Russian bot data
    files = get_data_files()
    dfs = []
    total_loaded = 0
    
    for file_path in files:
        if total_loaded >= max_samples:
            break
            
        df = pd.read_csv(file_path)
        
        # Filter for English tweets from Russian accounts
        english_df = df[
            (df['language'] == 'English') & 
            (df['author'].notna()) & 
            (df['content'].notna()) &
            (df['content'].str.len() > 20)  # Minimum length
        ].copy()
        
        if len(english_df) > 0:
            english_df['label'] = 1  # Russian bot = 1
            english_df['source'] = 'real_russian'
            dfs.append(english_df[['content', 'label', 'source']].rename(columns={'content': 'text'}))
            total_loaded += len(english_df)
            print(f"   Loaded {len(english_df):,} English tweets from {file_path}")
            
        if total_loaded >= max_samples:
            break
    
    if not dfs:
        print("⚠️ No Russian bot data found, using synthetic data only")
        real_df = pd.DataFrame()
    else:
        real_df = pd.concat(dfs, ignore_index=True)
        # Sample to manage size
        if len(real_df) > max_samples // 2:
            real_df = real_df.sample(n=max_samples // 2, random_state=42)
    
    # Generate synthetic data
    extractor = EnhancedRussianLinguisticExtractor()
    synthetic_df = extractor.generate_synthetic_data(n_samples=max_samples // 2)
    
    # Combine real and synthetic data
    if len(real_df) > 0:
        combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
    else:
        combined_df = synthetic_df
    
    print(f"📊 Dataset Summary:")
    print(f"   Real Russian bot tweets: {len(real_df):,}")
    print(f"   Synthetic samples: {len(synthetic_df):,}")
    print(f"   Total samples: {len(combined_df):,}")
    print(f"   Russian samples: {sum(combined_df['label'] == 1):,}")
    print(f"   Control samples: {sum(combined_df['label'] == 0):,}")
    
    return combined_df

def extract_features_from_data(df):
    """Extract features from the dataset"""
    print("\n🔍 Extracting linguistic features...")
    
    extractor = EnhancedRussianLinguisticExtractor()
    features_list = []
    
    for idx, row in df.iterrows():
        features = extractor.extract_features(row['text'])
        features['label'] = row['label']
        features['source'] = row['source']
        features_list.append(features)
        
        if (idx + 1) % 1000 == 0:
            print(f"   Processed {idx + 1:,} samples...")
    
    features_df = pd.DataFrame(features_list)
    print(f"   ✅ Extracted {len(features_df.columns) - 2} features from {len(features_df):,} samples")
    
    return features_df

def bootstrap_validation(X, y, model_class, n_bootstrap=100, test_size=0.3, **model_params):
    """Perform bootstrap validation to get robust performance estimates"""
    print(f"\n🔄 Running bootstrap validation with {n_bootstrap} iterations...")
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    auc_scores = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        X_boot, y_boot = resample(X, y, random_state=i)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_boot, y_boot, test_size=test_size, random_state=i, stratify=y_boot
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = model_class(**model_params)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_pred_proba))
        
        if (i + 1) % 20 == 0:
            print(f"   Completed {i + 1}/{n_bootstrap} bootstrap iterations...")
    
    # Calculate confidence intervals
    def confidence_interval(scores, confidence=0.95):
        alpha = 1 - confidence
        lower = np.percentile(scores, (alpha/2) * 100)
        upper = np.percentile(scores, (1 - alpha/2) * 100)
        return np.mean(scores), lower, upper
    
    results = {
        'accuracy': confidence_interval(accuracies),
        'precision': confidence_interval(precisions),
        'recall': confidence_interval(recalls),
        'f1': confidence_interval(f1_scores),
        'auc': confidence_interval(auc_scores)
    }
    
    return results

def train_enhanced_models(features_df):
    """Train enhanced models with high sensitivity"""
    print("\n🤖 Training enhanced Russian linguistic detector...")
    
    # Prepare features and labels
    feature_cols = [col for col in features_df.columns if col not in ['label', 'source']]
    X = features_df[feature_cols]
    y = features_df['label']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression with high sensitivity (low threshold)
    # Adjust class weights to favor detecting Russian patterns
    lr_model = LogisticRegression(
        random_state=42, 
        max_iter=2000,
        class_weight={0: 1, 1: 2}  # Higher weight for Russian class
    )
    lr_model.fit(X_train_scaled, y_train)
    
    # Train Random Forest with high sensitivity
    rf_model = RandomForestClassifier(
        n_estimators=200, 
        random_state=42,
        class_weight={0: 1, 1: 2},  # Higher weight for Russian class
        min_samples_split=2,
        min_samples_leaf=1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate with different thresholds for sensitivity
    print("\n📊 Model Performance (High Sensitivity):")
    
    models = [("Logistic Regression", lr_model), ("Random Forest", rf_model)]
    
    for name, model in models:
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Test different thresholds
        for threshold in [0.3, 0.4, 0.5]:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred_thresh)
            precision = precision_score(y_test, y_pred_thresh)
            recall = recall_score(y_test, y_pred_thresh)
            f1 = f1_score(y_test, y_pred_thresh)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"   {name} (threshold={threshold}):")
            print(f"     Accuracy: {accuracy:.3f}")
            print(f"     Precision: {precision:.3f}")
            print(f"     Recall: {recall:.3f}")
            print(f"     F1-Score: {f1:.3f}")
            print(f"     AUC: {auc:.3f}")
            print()
    
    # Bootstrap validation for Logistic Regression
    print("🎯 Bootstrap Validation Results:")
    bootstrap_results = bootstrap_validation(
        X, y, LogisticRegression, 
        n_bootstrap=50,
        random_state=42, 
        max_iter=2000,
        class_weight={0: 1, 1: 2}
    )
    
    for metric, (mean, lower, upper) in bootstrap_results.items():
        print(f"   {metric.capitalize()}: {mean:.3f} (95% CI: {lower:.3f}-{upper:.3f})")
    
    return lr_model, rf_model, scaler, feature_cols, X_test, y_test, bootstrap_results

def analyze_pattern_importance(model, feature_cols, features_df):
    """Analyze which patterns are most important for detection"""
    print("\n🔍 ENHANCED PATTERN ANALYSIS")
    print("=" * 50)
    
    # Get feature importance
    coefficients = model.coef_[0]
    feature_importance = list(zip(feature_cols, coefficients))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\n🏆 TOP SENSITIVITY INDICATORS:")
    print("(Positive = Russian-like, Negative = English-like)")
    for i, (feature, coef) in enumerate(feature_importance[:20]):
        direction = "🚨" if coef > 0 else "✅"
        print(f"   {i+1:2d}. {direction} {feature:30} = {coef:+.3f}")
    
    # Analyze rule-based accuracy
    rule_based_acc = (features_df['rule_based_russian'] == features_df['label']).mean()
    print(f"\n🎯 Rule-based Classification Accuracy: {rule_based_acc:.3f}")
    
    # Pattern frequency analysis
    print(f"\n📊 PATTERN FREQUENCY IN DATASET:")
    pattern_cols = ['critical_russian_patterns', 'moderate_russian_patterns', 'weak_russian_patterns']
    
    for col in pattern_cols:
        if col in features_df.columns:
            russian_mean = features_df[features_df['label'] == 1][col].mean()
            english_mean = features_df[features_df['label'] == 0][col].mean()
            
            print(f"   {col:30} Russian: {russian_mean:.3f}, English: {english_mean:.3f}")

def create_enhanced_visualizations(lr_model, rf_model, X_test, y_test, feature_cols, bootstrap_results):
    """Create enhanced visualizations"""
    print("\n📈 Creating enhanced analysis visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Feature Importance
    coefficients = lr_model.coef_[0]
    top_features_idx = np.argsort(np.abs(coefficients))[-15:]
    
    colors = ['red' if coefficients[i] > 0 else 'blue' for i in top_features_idx]
    axes[0,0].barh(range(len(top_features_idx)), coefficients[top_features_idx], color=colors)
    axes[0,0].set_yticks(range(len(top_features_idx)))
    axes[0,0].set_yticklabels([feature_cols[i] for i in top_features_idx], fontsize=8)
    axes[0,0].set_title('Enhanced Linguistic Features\n(Red=Russian, Blue=English)')
    axes[0,0].set_xlabel('Coefficient Value')
    
    # 2. Sensitivity Analysis
    thresholds = np.linspace(0.1, 0.9, 50)
    precisions = []
    recalls = []
    
    y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        precisions.append(precision_score(y_test, y_pred_thresh))
        recalls.append(recall_score(y_test, y_pred_thresh))
    
    axes[0,1].plot(thresholds, precisions, label='Precision', linewidth=2)
    axes[0,1].plot(thresholds, recalls, label='Recall', linewidth=2)
    axes[0,1].axvline(x=0.3, color='red', linestyle='--', alpha=0.7, label='High Sensitivity')
    axes[0,1].set_xlabel('Classification Threshold')
    axes[0,1].set_ylabel('Score')
    axes[0,1].set_title('Precision-Recall vs Threshold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Bootstrap Results
    metrics = list(bootstrap_results.keys())
    means = [bootstrap_results[m][0] for m in metrics]
    errors = [(bootstrap_results[m][0] - bootstrap_results[m][1], 
               bootstrap_results[m][2] - bootstrap_results[m][0]) for m in metrics]
    
    axes[0,2].bar(range(len(metrics)), means, yerr=np.array(errors).T, capsize=5)
    axes[0,2].set_xticks(range(len(metrics)))
    axes[0,2].set_xticklabels(metrics, rotation=45)
    axes[0,2].set_title('Bootstrap Validation Results\n(with 95% Confidence Intervals)')
    axes[0,2].set_ylabel('Score')
    
    # 4. Pattern Distribution
    pattern_cols = ['critical_russian_patterns', 'moderate_russian_patterns', 'weighted_russian_score']
    
    for i, col in enumerate(pattern_cols):
        if col in feature_cols:
            idx = feature_cols.index(col)
            col_data = X_test.iloc[:, idx]
            
            axes[1,i].hist(col_data[y_test == 0], bins=20, alpha=0.7, label='English', color='skyblue')
            axes[1,i].hist(col_data[y_test == 1], bins=20, alpha=0.7, label='Russian', color='lightcoral')
            axes[1,i].set_title(f'Distribution: {col}')
            axes[1,i].set_xlabel('Pattern Count')
            axes[1,i].set_ylabel('Frequency')
            axes[1,i].legend()
    
    plt.tight_layout()
    plt.savefig('enhanced_russian_linguistic_analysis.png', dpi=300, bbox_inches='tight')
    print("   📊 Saved enhanced visualization to 'enhanced_russian_linguistic_analysis.png'")

def demonstrate_enhanced_detection(lr_model, scaler, feature_cols):
    """Demonstrate the enhanced detector with high sensitivity"""
    print("\n🎯 ENHANCED DETECTION DEMONSTRATION")
    print("=" * 60)
    
    extractor = EnhancedRussianLinguisticExtractor()
    
    test_samples = [
        {
            'name': 'Perfect English',
            'text': "I'm really excited about the upcoming movie. The trailer looks fantastic!"
        },
        {
            'name': 'One Critical Error',
            'text': "This situation is very actual for our country today."
        },
        {
            'name': 'Two Moderate Errors', 
            'text': "I am knowing that internet has many informations about this topic."
        },
        {
            'name': 'Multiple Minor Errors',
            'text': "russian people make sport very well. they have good results in different competitions."
        },
        {
            'name': 'Heavy Russian Influence',
            'text': "Very actual problem in america depends from government decision. People must take decision about this situation."
        }
    ]
    
    print("🔍 Testing Enhanced Sensitivity (1-2 patterns = Russian classification):")
    
    for sample in test_samples:
        features = extractor.extract_features(sample['text'])
        features_array = np.array([features[col] for col in feature_cols]).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        russian_prob = lr_model.predict_proba(features_scaled)[0, 1]
        
        # Show pattern breakdown
        critical = features['critical_russian_patterns']
        moderate = features['moderate_russian_patterns'] 
        weak = features['weak_russian_patterns']
        rule_based = features['rule_based_russian']
        
        print(f"\n📝 {sample['name']}:")
        print(f"   Text: \"{sample['text']}\"")
        print(f"   🎯 Russian Probability: {russian_prob:.1%}")
        print(f"   🚨 Critical patterns: {critical}")
        print(f"   ⚠️  Moderate patterns: {moderate}")
        print(f"   💡 Weak patterns: {weak}")
        print(f"   🤖 Rule-based classification: {'RUSSIAN' if rule_based else 'ENGLISH'}")
        
        # High sensitivity classification (threshold = 0.3)
        classification = "🇷🇺 RUSSIAN" if russian_prob >= 0.3 else "🇺🇸 ENGLISH"
        print(f"   📊 Enhanced Classification: {classification}")

def save_enhanced_models(lr_model, rf_model, scaler, feature_cols, bootstrap_results):
    """Save the enhanced models and results"""
    print("\n💾 Saving enhanced linguistic detection models...")
    
    joblib.dump(lr_model, MODELS_DIR / "enhanced_russian_linguistic_lr.pkl")
    joblib.dump(rf_model, MODELS_DIR / "enhanced_russian_linguistic_rf.pkl")
    joblib.dump(scaler, MODELS_DIR / "enhanced_russian_linguistic_scaler.pkl")
    
    with open(MODELS_DIR / "enhanced_russian_features.txt", 'w') as f:
        for feature in feature_cols:
            f.write(f"{feature}\n")
    
    # Save bootstrap results
    bootstrap_df = pd.DataFrame(bootstrap_results)
    bootstrap_df.to_csv(MODELS_DIR / "bootstrap_validation_results.csv")
    
    print(f"   ✅ Enhanced models saved to {MODELS_DIR}/")

def main():
    """Main function for enhanced Russian linguistic detection"""
    print("🇷🇺 ENHANCED RUSSIAN LINGUISTIC DETECTOR")
    print("=" * 60)
    print("High-sensitivity detection: 1-2 infractions = Russian classification")
    print("Includes bootstrapping validation for robust performance estimates")
    print()
    
    try:
        # Load and prepare data
        df = load_and_prepare_data(max_samples=20000)
        
        # Extract features
        features_df = extract_features_from_data(df)
        
        # Train enhanced models
        lr_model, rf_model, scaler, feature_cols, X_test, y_test, bootstrap_results = train_enhanced_models(features_df)
        
        # Analyze patterns
        analyze_pattern_importance(lr_model, feature_cols, features_df)
        
        # Create visualizations
        create_enhanced_visualizations(lr_model, rf_model, X_test, y_test, feature_cols, bootstrap_results)
        
        # Demonstrate detection
        demonstrate_enhanced_detection(lr_model, scaler, feature_cols)
        
        # Save models
        save_enhanced_models(lr_model, rf_model, scaler, feature_cols, bootstrap_results)
        
        print(f"\n✅ SUCCESS! Enhanced Russian linguistic detector created")
        print(f"\n🎯 ENHANCED SUMMARY:")
        print(f"   • High sensitivity: 1 critical OR 2 moderate patterns = Russian")
        print(f"   • Bootstrap validation with 50 iterations")
        print(f"   • Analyzed {len(df):,} samples ({len(features_df.columns)-2} features)")
        
        # Final accuracy summary
        print(f"\n📊 FINAL PERFORMANCE ESTIMATES:")
        for metric, (mean, lower, upper) in bootstrap_results.items():
            print(f"   {metric.capitalize()}: {mean:.1%} (95% CI: {lower:.1%}-{upper:.1%})")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 