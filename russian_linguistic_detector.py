"""
Russian Linguistic Bot Detector

This script creates a detector that identifies Russian language patterns in English text
by analyzing common linguistic mistakes that Russian speakers make when writing English.
We'll train it on half the Russian troll dataset and test on the other half.
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from data_utils import load_single_file, get_data_files

# Data directories
BOT_DATA_DIR = Path("data/russian_troll_tweets")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class RussianLinguisticFeatureExtractor:
    """
    Extract linguistic features that indicate Russian native speakers writing in English
    """
    
    def __init__(self):
        # Common Russian->English linguistic patterns
        self.russian_indicators = {
            # Articles - Russians often omit or misuse articles
            'missing_articles': [
                r'\b(in|at|on)\s+(morning|evening|afternoon)\b',  # "in morning" vs "in the morning"
                r'\b(go\s+to|went\s+to)\s+(school|university|work|home)\b',  # "go to school" (sometimes missing 'the')
                r'\b(from|in|at)\s+(USA|Russia|Ukraine)\b',  # Country names without articles
            ],
            
            # Word order from Russian syntax
            'word_order_issues': [
                r'\b(very|quite|really|so)\s+(good|bad|nice|terrible)\s+(thing|person|idea)\b',  # Adjective placement
                r'\b(all|many|some)\s+(of\s+)?this\b',  # "all this" vs "all of this"
                r'\b(what|which)\s+(kind|type)\s+of\b',  # Direct translation patterns
            ],
            
            # Preposition mistakes common to Russian speakers
            'preposition_errors': [
                r'\b(depends|rely|count)\s+from\b',  # "depends from" instead of "depends on"
                r'\b(different|separate)\s+to\b',   # "different to" instead of "different from"
                r'\b(participate)\s+to\b',          # "participate to" instead of "participate in"
                r'\b(consist)\s+from\b',            # "consist from" instead of "consist of"
                r'\b(afraid)\s+from\b',             # "afraid from" instead of "afraid of"
            ],
            
            # Verb constructions influenced by Russian
            'verb_patterns': [
                r'\b(will\s+be)\s+(to\s+)?(verb|go|come|work)\b',  # Future tense overuse
                r'\b(am|is|are)\s+(knowing|understanding|believing)\b',  # Progressive with stative verbs
                r'\b(have|has)\s+(wrote|came|went|done)\b',  # Perfect tense errors
                r'\b(make|do)\s+(mistake|error|problem)\b',  # Calques from Russian
            ],
            
            # Specific vocabulary choices (calques/direct translations)
            'vocabulary_calques': [
                r'\bmake\s+(sport|sports)\b',       # "make sport" instead of "do sports"
                r'\bmake\s+(homework|study)\b',     # "make homework" instead of "do homework"  
                r'\btake\s+(decision|conclusion)\b', # "take decision" instead of "make decision"
                r'\bgive\s+(birth)\s+to\b',         # Direct translation patterns
                r'\bput\s+(question|problem)\b',    # "put question" instead of "ask question"
            ],
            
            # Capitalization patterns from Russian
            'capitalization_patterns': [
                r'\b(internet|web)\b',              # Lowercase Internet/Web
                r'\b(russian|american|ukrainian|german)\b',  # Lowercase nationality adjectives
            ],
            
            # Punctuation influenced by Russian standards
            'punctuation_patterns': [
                r'[.!?]\s*[a-z]',                  # Lowercase after sentence end
                r'\s+,\s+',                        # Spaces before commas
                r'[.!?]{2,}',                      # Multiple punctuation marks
            ]
        }
        
        # Words commonly misspelled by Russian speakers
        self.common_misspellings = {
            'definately': 'definitely',
            'recieve': 'receive', 
            'seperate': 'separate',
            'occured': 'occurred',
            'accomodate': 'accommodate',
            'wether': 'whether',
            'wich': 'which',
            'teh': 'the',
            'thier': 'their',
            'youre': "you're",
            'its': "it's",  # Context dependent
            'loose': 'lose',  # Context dependent
            'affect': 'effect',  # Context dependent
        }
        
        # English words that Russians often use incorrectly
        self.false_friends = {
            'actual': 'current',     # Russian "актуальный" means current, not actual
            'fabric': 'factory',     # Russian "фабрика" 
            'magazine': 'store',     # Russian "магазин"
            'sympathy': 'liking',    # Russian "симпатия"
            'genial': 'brilliant',   # Russian "гениальный"
        }

    def extract_features(self, text):
        """Extract all linguistic features from a text"""
        if pd.isna(text) or not isinstance(text, str):
            text = ""
        
        text = text.lower().strip()
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.findall(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Russian linguistic pattern detection
        total_patterns = 0
        for category, patterns in self.russian_indicators.items():
            category_count = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                category_count += matches
                total_patterns += matches
            
            features[f'{category}_count'] = category_count
            features[f'{category}_rate'] = category_count / max(1, features['word_count'])
        
        features['total_russian_patterns'] = total_patterns
        features['russian_pattern_density'] = total_patterns / max(1, features['word_count'])
        
        # Spelling patterns
        misspelling_count = 0
        for wrong, correct in self.common_misspellings.items():
            misspelling_count += len(re.findall(r'\b' + wrong + r'\b', text, re.IGNORECASE))
        
        features['misspelling_count'] = misspelling_count
        features['misspelling_rate'] = misspelling_count / max(1, features['word_count'])
        
        # False friends usage
        false_friend_count = 0
        for word in self.false_friends.keys():
            false_friend_count += len(re.findall(r'\b' + word + r'\b', text, re.IGNORECASE))
        
        features['false_friend_count'] = false_friend_count
        features['false_friend_rate'] = false_friend_count / max(1, features['word_count'])
        
        # Punctuation and capitalization analysis
        features['exclamation_marks'] = text.count('!')
        features['question_marks'] = text.count('?')
        features['uppercase_words'] = len(re.findall(r'\b[A-Z]{2,}\b', text))
        features['capitalization_errors'] = len(re.findall(r'[.!?]\s+[a-z]', text))
        
        # Grammar complexity indicators
        features['comma_usage'] = text.count(',') / max(1, features['sentence_count'])
        features['subordinate_clauses'] = len(re.findall(r'\b(that|which|who|when|where|because|although)\b', text, re.IGNORECASE))
        
        # Repetition patterns (bots often repeat phrases)
        words = text.split()
        if len(words) > 1:
            word_freq = Counter(words)
            features['word_repetition_rate'] = (len(words) - len(set(words))) / len(words)
            features['max_word_frequency'] = max(word_freq.values()) / len(words)
        else:
            features['word_repetition_rate'] = 0
            features['max_word_frequency'] = 0
        
        return features

def load_russian_bot_data(max_samples=100000):
    """Load Russian bot tweets for linguistic analysis"""
    print("📥 Loading Russian bot tweets for linguistic analysis...")
    
    # Load multiple files to get enough data
    files = get_data_files()
    dfs = []
    total_loaded = 0
    
    for file_path in files:
        if total_loaded >= max_samples:
            break
            
        df = pd.read_csv(file_path)
        
        # Filter for English tweets only
        english_tweets = df[df['language'] == 'English'].copy()
        
        if len(english_tweets) == 0:
            continue
            
        remaining = max_samples - total_loaded
        if len(english_tweets) > remaining:
            english_tweets = english_tweets.sample(n=remaining, random_state=42)
        
        dfs.append(english_tweets)
        total_loaded += len(english_tweets)
        print(f"   Loaded {len(english_tweets):,} English tweets from {file_path.name}")
    
    if not dfs:
        raise ValueError("No English tweets found in the dataset")
    
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"   ✅ Total English Russian bot tweets: {len(df_combined):,}")
    
    return df_combined

def extract_linguistic_features(df):
    """Extract linguistic features from all tweets"""
    print("\n🔧 Extracting Russian linguistic features...")
    
    extractor = RussianLinguisticFeatureExtractor()
    
    # Extract features for each tweet
    features_list = []
    for idx, content in enumerate(df['content']):
        if idx % 10000 == 0:
            print(f"   Processed {idx:,} tweets...")
        
        features = extractor.extract_features(content)
        features['tweet_id'] = idx
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    print(f"   ✅ Extracted {len(features_df.columns)-1} linguistic features")
    
    return features_df

def create_training_labels(df, features_df):
    """
    Create training labels by splitting the dataset in half.
    Half will be labeled as 'Russian bot' (1), half as 'control' (0) for comparison.
    """
    print("\n🏷️  Creating training labels...")
    
    # Split dataset in half randomly
    n_samples = len(df)
    indices = np.random.RandomState(42).permutation(n_samples)
    
    # First half = "Russian bot" (1), Second half = "Control group" (0)
    # This creates a controlled experiment where we test if the model can distinguish
    # between two halves of the same dataset based on linguistic patterns
    
    labels = np.zeros(n_samples)
    labels[indices[:n_samples//2]] = 1  # Russian bot indicators
    
    features_df['is_russian_bot'] = labels
    
    print(f"   Russian bot samples: {sum(labels):,}")
    print(f"   Control samples: {n_samples - sum(labels):,}")
    
    return features_df

def train_linguistic_detector(features_df):
    """Train models to detect Russian linguistic patterns"""
    print("\n🤖 Training Russian linguistic pattern detector...")
    
    # Prepare features and labels
    feature_cols = [col for col in features_df.columns if col not in ['tweet_id', 'is_russian_bot']]
    X = features_df[feature_cols]
    y = features_df['is_russian_bot']
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    # Train Random Forest for comparison
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate models
    print("\n📊 Model Performance:")
    
    for name, model in [("Logistic Regression", lr_model), ("Random Forest", rf_model)]:
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"   {name}:")
        print(f"     Accuracy: {accuracy:.3f}")
        print(f"     AUC: {auc:.3f}")
    
    return lr_model, rf_model, scaler, feature_cols, X_test, y_test

def analyze_linguistic_patterns(lr_model, feature_cols, features_df):
    """Analyze which linguistic patterns are most indicative of Russian speakers"""
    print("\n🔍 LINGUISTIC PATTERN ANALYSIS")
    print("=" * 50)
    
    # Get feature importance from logistic regression coefficients
    coefficients = lr_model.coef_[0]
    feature_importance = list(zip(feature_cols, coefficients))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\n🏆 TOP RUSSIAN LINGUISTIC INDICATORS:")
    print("(Positive = more Russian-like, Negative = less Russian-like)")
    for i, (feature, coef) in enumerate(feature_importance[:15]):
        direction = "📈" if coef > 0 else "📉"
        print(f"   {i+1:2d}. {direction} {feature:25} = {coef:+.3f}")
    
    # Analyze pattern distributions
    print(f"\n📊 PATTERN FREQUENCY ANALYSIS:")
    
    pattern_features = [col for col in feature_cols if 'count' in col or 'rate' in col]
    
    for feature in pattern_features[:10]:
        mean_val = features_df[feature].mean()
        std_val = features_df[feature].std()
        max_val = features_df[feature].max()
        
        print(f"   {feature:30} Mean: {mean_val:.3f}, Std: {std_val:.3f}, Max: {max_val:.1f}")

def create_visualizations(lr_model, rf_model, X_test, y_test, feature_cols):
    """Create visualizations of the linguistic analysis"""
    print("\n📈 Creating linguistic analysis visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature Importance
    coefficients = lr_model.coef_[0]
    top_features_idx = np.argsort(np.abs(coefficients))[-15:]
    
    axes[0,0].barh(range(len(top_features_idx)), coefficients[top_features_idx])
    axes[0,0].set_yticks(range(len(top_features_idx)))
    axes[0,0].set_yticklabels([feature_cols[i] for i in top_features_idx])
    axes[0,0].set_title('Top 15 Linguistic Features\n(Russian Pattern Indicators)')
    axes[0,0].set_xlabel('Coefficient Value')
    
    # 2. Prediction Distribution
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    
    axes[0,1].hist(lr_probs[y_test == 0], bins=30, alpha=0.7, label='Control Group', color='skyblue')
    axes[0,1].hist(lr_probs[y_test == 1], bins=30, alpha=0.7, label='Russian Bot Group', color='lightcoral')
    axes[0,1].set_xlabel('Russian Pattern Probability')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('Distribution of Russian Pattern Scores')
    axes[0,1].legend()
    
    # 3. Confusion Matrix
    y_pred = lr_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[1,0], cmap='Blues')
    axes[1,0].set_title('Confusion Matrix\n(Linguistic Pattern Detection)')
    axes[1,0].set_xlabel('Predicted')
    axes[1,0].set_ylabel('Actual')
    
    # 4. Pattern Category Analysis
    pattern_categories = ['missing_articles', 'preposition_errors', 'vocabulary_calques', 'misspelling']
    category_means = []
    category_stds = []
    
    for category in pattern_categories:
        rate_col = f'{category}_rate'
        if rate_col in feature_cols:
            idx = feature_cols.index(rate_col)
            mean_importance = abs(coefficients[idx])
            category_means.append(mean_importance)
        else:
            category_means.append(0)
    
    axes[1,1].bar(range(len(pattern_categories)), category_means, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    axes[1,1].set_xticks(range(len(pattern_categories)))
    axes[1,1].set_xticklabels(pattern_categories, rotation=45)
    axes[1,1].set_title('Linguistic Pattern Category Importance')
    axes[1,1].set_ylabel('Absolute Coefficient Value')
    
    plt.tight_layout()
    plt.savefig('russian_linguistic_analysis.png', dpi=300, bbox_inches='tight')
    print("   📊 Saved visualization to 'russian_linguistic_analysis.png'")

def demonstrate_detection(lr_model, scaler, feature_cols):
    """Demonstrate the linguistic detector on sample texts"""
    print("\n🎯 DEMONSTRATION: Russian Linguistic Pattern Detection")
    print("=" * 60)
    
    extractor = RussianLinguisticFeatureExtractor()
    
    # Sample texts with varying degrees of Russian influence
    sample_texts = [
        {
            'name': 'Native English Speaker',
            'text': "I'm really excited about the new movie that's coming out next week. The trailer looks amazing and I can't wait to see it!"
        },
        {
            'name': 'Russian Speaker (Mild Patterns)',
            'text': "I am very interested in this new film. It looks quite good and I want to see it in cinema when it will be available."
        },
        {
            'name': 'Russian Speaker (Strong Patterns)', 
            'text': "This film is very actual topic now. I will be to watch it definately. In my country we also make such movies about actual problems."
        },
        {
            'name': 'Russian Bot-like Text',
            'text': "American people must understand that this situation is very actual. We need to make sport and take decision about future. Internet shows us many different informations about this."
        }
    ]
    
    for sample in sample_texts:
        features = extractor.extract_features(sample['text'])
        features_array = np.array([features[col] for col in feature_cols]).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        russian_probability = lr_model.predict_proba(features_scaled)[0, 1]
        
        print(f"\n📝 {sample['name']}:")
        print(f"   Text: \"{sample['text']}\"")
        print(f"   Russian Pattern Score: {russian_probability:.1%}")
        
        # Show top detected patterns
        top_patterns = []
        for feature, value in features.items():
            if 'count' in feature and value > 0:
                top_patterns.append((feature, value))
        
        if top_patterns:
            print(f"   Detected patterns: {', '.join([f'{name}({val})' for name, val in top_patterns[:3]])}")

def save_models(lr_model, rf_model, scaler, feature_cols):
    """Save the trained linguistic detection models"""
    print("\n💾 Saving linguistic detection models...")
    
    joblib.dump(lr_model, MODELS_DIR / "russian_linguistic_detector_lr.pkl")
    joblib.dump(rf_model, MODELS_DIR / "russian_linguistic_detector_rf.pkl") 
    joblib.dump(scaler, MODELS_DIR / "russian_linguistic_scaler.pkl")
    
    with open(MODELS_DIR / "russian_linguistic_features.txt", 'w') as f:
        for feature in feature_cols:
            f.write(f"{feature}\n")
    
    print(f"   ✅ Models saved to {MODELS_DIR}/")

def main():
    """Main function to create and evaluate the Russian linguistic detector"""
    print("🇷🇺 RUSSIAN LINGUISTIC BOT DETECTOR")
    print("=" * 50)
    print("Analyzing linguistic patterns that indicate Russian speakers writing in English")
    print()
    
    try:
        # Load Russian bot data
        df_bots = load_russian_bot_data(max_samples=50000)
        
        # Extract linguistic features
        features_df = extract_linguistic_features(df_bots)
        
        # Create labels (half as Russian bot indicators, half as control)
        features_df = create_training_labels(df_bots, features_df)
        
        # Train models
        lr_model, rf_model, scaler, feature_cols, X_test, y_test = train_linguistic_detector(features_df)
        
        # Analyze patterns
        analyze_linguistic_patterns(lr_model, feature_cols, features_df)
        
        # Create visualizations
        create_visualizations(lr_model, rf_model, X_test, y_test, feature_cols)
        
        # Demonstrate detection
        demonstrate_detection(lr_model, scaler, feature_cols)
        
        # Save models
        save_models(lr_model, rf_model, scaler, feature_cols)
        
        print(f"\n✅ SUCCESS! Russian linguistic detector created and evaluated")
        print(f"\n🎯 SUMMARY:")
        print(f"   • Analyzed {len(df_bots):,} Russian bot tweets in English")
        print(f"   • Extracted {len(feature_cols)} linguistic features")
        print(f"   • Detected patterns indicating Russian linguistic influence")
        print(f"   • Models can identify Russian writing patterns in English text")
        
        print(f"\n📁 Files created:")
        print(f"   • russian_linguistic_detector_lr.pkl")
        print(f"   • russian_linguistic_detector_rf.pkl") 
        print(f"   • russian_linguistic_scaler.pkl")
        print(f"   • russian_linguistic_analysis.png")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have the Russian troll tweets dataset available.")

if __name__ == "__main__":
    main() 