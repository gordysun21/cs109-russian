"""
Enhanced Russian Linguistic Bot Detector

This enhanced version compares Russian bot tweets against legitimate human tweets
to identify linguistic patterns that specifically indicate Russian native speakers.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from data_utils import load_single_file, get_data_files

# Data directories
BOT_DATA_DIR = Path("data/russian_troll_tweets")
HUMAN_DATA_DIR = Path("data/legitimate_tweets")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class EnhancedRussianLinguisticExtractor:
    """
    Enhanced linguistic feature extractor with more sophisticated patterns
    """
    
    def __init__(self):
        # More specific Russian linguistic patterns
        self.russian_patterns = {
            # Article usage errors (Russians struggle with a/an/the)
            'article_errors': [
                r'\b(go|went|going)\s+to\s+(school|university|hospital|church|work)\b',  # Missing "the"
                r'\b(in|on|at)\s+(morning|evening|afternoon|night)\b',  # Missing "the"
                r'\b(play|playing)\s+(football|basketball|tennis|piano|guitar)\b',  # Missing "the"
                r'\b(from|in|to)\s+(Russia|USA|Ukraine|China|Germany)\b',  # Article with countries
                r'\b(internet|web)\s+(site|page|connection)\b',  # Missing "the Internet"
            ],
            
            # Preposition confusion (major Russian L2 error)
            'preposition_confusion': [
                r'\b(depends|depend|depending)\s+(from|of)\b',  # "depends from" not "depends on"
                r'\b(different|differ)\s+(to|with)\b',  # "different to" not "different from"
                r'\b(consist|consisting)\s+(from|with)\b',  # "consist from" not "consist of"
                r'\b(participate|participating)\s+(to|at)\b',  # "participate to" not "participate in"
                r'\b(afraid|scared)\s+(from|of)\b',  # "afraid from" not "afraid of"
                r'\b(angry|mad)\s+(on|to)\b',  # "angry on" not "angry at/with"
                r'\b(listen|listening)\s+(on)\b',  # "listen on" not "listen to"
            ],
            
            # False friends (Russian words that look like English but mean different things)
            'false_friends': [
                r'\b(actual|actually)\b',  # Russian "актуальный" = relevant, not actual
                r'\b(fabric)\b',  # Russian "фабрика" = factory, not fabric
                r'\b(magazine)\b',  # Russian "магазин" = store, not magazine  
                r'\b(family)\b',  # Russian "фамилия" = surname, not family
                r'\b(decoration)\b',  # Russian "декорация" = scenery, not decoration
                r'\b(accurate)\b',  # Often misused
                r'\b(sympathy)\b',  # Russian "симпатия" = liking, not sympathy
            ],
            
            # Calques (direct translations from Russian)
            'russian_calques': [
                r'\bmake\s+(sport|sports)\b',  # "заниматься спортом" = "make sport"
                r'\bmake\s+(photo|picture)\b',  # "сделать фото" = "make photo"
                r'\btake\s+(decision|conclusion)\b',  # "принять решение" = "take decision"
                r'\bput\s+(question|signature)\b',  # "поставить вопрос/подпись" = "put question/signature"
                r'\bgive\s+(birth)\s+to\b',  # Direct translation
                r'\bmake\s+(homework|lesson)\b',  # "делать урок" = "make lesson"
                r'\btake\s+(place|seat)\b',  # Sometimes incorrect usage
            ],
            
            # Word order influenced by Russian syntax
            'word_order_russian': [
                r'\b(very|quite|really)\s+(interesting|important|difficult)\s+(question|problem|situation)\b',
                r'\b(such|this)\s+(kind|type)\s+of\s+(people|things|problems)\b',
                r'\b(all|many)\s+(of\s+)?these\s+(people|things|problems)\b',
                r'\b(one|some)\s+of\s+(this|these)\b',  # "one of this" not "one of these"
            ],
            
            # Verb tense and aspect errors typical of Russian speakers
            'verb_errors': [
                r'\b(will\s+be)\s+(go|come|study|work)\b',  # Future tense confusion
                r'\b(am|is|are)\s+(knowing|understanding|meaning|belonging)\b',  # Stative verbs in progressive
                r'\b(have|has)\s+(came|went|seen|done|written)\b',  # Perfect tense formation errors
                r'\bwould\s+(like\s+to\s+)?go\b',  # Conditional overuse
            ],
            
            # Capitalization errors from Russian rules
            'capitalization_russian': [
                r'\b(russian|american|english|german|chinese|ukrainian)\b',  # Nationalities lowercase
                r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',  # Days lowercase
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',  # Months lowercase
                r'\b(internet|web)\b',  # Technology terms lowercase
            ],
            
            # Punctuation influenced by Russian standards  
            'punctuation_russian': [
                r'[а-яё]+',  # Cyrillic characters mixed in
                r'\s+,',  # Space before comma (Russian style)
                r'[.!?]{2,}',  # Multiple punctuation (emotional style)
                r'[.!?]\s*[a-z]',  # No capitalization after period
            ]
        }
        
        # Common spelling mistakes by Russian speakers
        self.russian_misspellings = {
            'recieve': 'receive',
            'seperate': 'separate', 
            'definately': 'definitely',
            'occured': 'occurred',
            'accomodate': 'accommodate',
            'wich': 'which',
            'wether': 'whether',
            'teh': 'the',
            'thier': 'their',
            'youre': "you're",
            'loose': 'lose',  # Context-dependent confusion
            'chose': 'choose',  # Tense confusion
            'advise': 'advice',  # Noun/verb confusion
        }
        
        # Words that Russians overuse or use inappropriately
        self.russian_overuse = [
            'very', 'quite', 'really', 'such', 'so', 'this', 'that',
            'of course', 'actually', 'maybe', 'perhaps', 'probably'
        ]

    def extract_enhanced_features(self, text):
        """Extract enhanced linguistic features focusing on Russian patterns"""
        if pd.isna(text) or not isinstance(text, str):
            text = ""
        
        original_text = text
        text = text.lower().strip()
        features = {}
        
        # Basic text statistics
        words = text.split()
        sentences = re.findall(r'[.!?]+', text)
        
        features['text_length'] = len(text)
        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['avg_sentence_length'] = len(words) / max(1, len(sentences))
        
        # Russian linguistic pattern detection
        total_russian_indicators = 0
        for category, patterns in self.russian_patterns.items():
            category_count = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                category_count += matches
                total_russian_indicators += matches
            
            features[f'{category}_count'] = category_count
            features[f'{category}_density'] = category_count / max(1, len(words))
        
        features['total_russian_indicators'] = total_russian_indicators
        features['russian_density'] = total_russian_indicators / max(1, len(words))
        
        # Spelling analysis
        misspelling_count = 0
        for wrong_spell in self.russian_misspellings.keys():
            misspelling_count += len(re.findall(r'\b' + wrong_spell + r'\b', text, re.IGNORECASE))
        
        features['misspelling_count'] = misspelling_count
        features['misspelling_density'] = misspelling_count / max(1, len(words))
        
        # Overuse patterns
        overuse_count = 0
        for overused_word in self.russian_overuse:
            overuse_count += len(re.findall(r'\b' + overused_word + r'\b', text, re.IGNORECASE))
        
        features['overuse_count'] = overuse_count
        features['overuse_density'] = overuse_count / max(1, len(words))
        
        # Grammar complexity (Russians often use simpler constructions)
        features['complex_sentences'] = len(re.findall(r'\b(although|however|nevertheless|moreover|furthermore)\b', text, re.IGNORECASE))
        features['subordinate_clauses'] = len(re.findall(r'\b(that|which|who|when|where|because|since|while)\b', text, re.IGNORECASE))
        features['passive_voice'] = len(re.findall(r'\b(was|were|is|are|been)\s+\w+ed\b', text, re.IGNORECASE))
        
        # Punctuation analysis
        features['exclamation_marks'] = text.count('!')
        features['question_marks'] = text.count('?')
        features['commas_per_sentence'] = text.count(',') / max(1, len(sentences))
        features['semicolons'] = text.count(';')
        features['quotation_marks'] = text.count('"') + text.count("'")
        
        # Capitalization patterns
        features['all_caps_words'] = len(re.findall(r'\b[A-Z]{2,}\b', original_text))
        features['capitalization_errors'] = len(re.findall(r'[.!?]\s+[a-z]', original_text))
        
        # Repetition and coherence (bots often have repetitive patterns)
        if len(words) > 1:
            word_freq = Counter(words)
            features['unique_word_ratio'] = len(set(words)) / len(words)
            features['most_frequent_word_ratio'] = max(word_freq.values()) / len(words)
            features['repetitive_words'] = sum(1 for count in word_freq.values() if count > 1)
        else:
            features['unique_word_ratio'] = 1.0
            features['most_frequent_word_ratio'] = 1.0
            features['repetitive_words'] = 0
        
        return features

def load_comparison_data(n_russian=25000, n_human=25000):
    """Load Russian bot tweets and human tweets for comparison"""
    print("📥 Loading data for Russian vs Human comparison...")
    
    # Load Russian bot tweets (English only)
    print("   Loading Russian bot tweets...")
    files = get_data_files()
    russian_tweets = []
    total_loaded = 0
    
    for file_path in files:
        if total_loaded >= n_russian:
            break
            
        df = pd.read_csv(file_path)
        english_tweets = df[df['language'] == 'English'].copy()
        
        if len(english_tweets) == 0:
            continue
            
        remaining = n_russian - total_loaded
        if len(english_tweets) > remaining:
            english_tweets = english_tweets.sample(n=remaining, random_state=42)
        
        russian_tweets.append(english_tweets[['content']])
        total_loaded += len(english_tweets)
        print(f"      Loaded {len(english_tweets):,} Russian tweets from {file_path.name}")
    
    df_russian = pd.concat(russian_tweets, ignore_index=True)
    df_russian['is_russian'] = 1
    df_russian['source'] = 'russian_bot'
    
    # Load human tweets
    print("   Loading human tweets...")
    human_files = list(HUMAN_DATA_DIR.glob("*.csv"))
    human_tweets = []
    total_human = 0
    
    for csv_file in human_files:
        if total_human >= n_human:
            break
            
        try:
            df = pd.read_csv(csv_file)
            remaining = n_human - total_human
            if len(df) > remaining:
                df = df.sample(n=remaining, random_state=42)
            
            if 'content' in df.columns:
                human_tweets.append(df[['content']])
                total_human += len(df)
                print(f"      Loaded {len(df):,} human tweets from {csv_file.name}")
        except Exception as e:
            print(f"      ⚠️  Error loading {csv_file.name}: {e}")
    
    if not human_tweets:
        raise ValueError("No human tweets found. Run download_legitimate_tweets.py first")
    
    df_human = pd.concat(human_tweets, ignore_index=True)
    df_human['is_russian'] = 0
    df_human['source'] = 'human'
    
    # Combine datasets
    df_combined = pd.concat([df_russian, df_human], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   ✅ Combined dataset: {len(df_russian):,} Russian + {len(df_human):,} Human = {len(df_combined):,} total")
    
    return df_combined

def extract_enhanced_features_batch(df):
    """Extract enhanced features from all tweets"""
    print("\n🔧 Extracting enhanced Russian linguistic features...")
    
    extractor = EnhancedRussianLinguisticExtractor()
    features_list = []
    
    for idx, content in enumerate(df['content']):
        if idx % 5000 == 0:
            print(f"   Processed {idx:,} tweets...")
        
        features = extractor.extract_enhanced_features(content)
        features['tweet_id'] = idx
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Add the labels
    features_df['is_russian'] = df['is_russian'].values
    features_df['source'] = df['source'].values
    
    print(f"   ✅ Extracted {len(features_df.columns)-3} enhanced linguistic features")
    
    return features_df

def train_enhanced_detector(features_df):
    """Train enhanced Russian vs Human detector"""
    print("\n🤖 Training Enhanced Russian vs Human Detector...")
    
    # Prepare features
    feature_cols = [col for col in features_df.columns 
                   if col not in ['tweet_id', 'is_russian', 'source']]
    
    X = features_df[feature_cols]
    y = features_df['is_russian']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Testing: {len(X_test):,} samples")
    print(f"   Russian samples in train: {y_train.sum():,}")
    print(f"   Human samples in train: {(1-y_train).sum():,}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    lr_model.fit(X_train_scaled, y_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\n📊 Enhanced Model Performance:")
    
    for name, model in [("Logistic Regression", lr_model), ("Random Forest", rf_model)]:
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"   {name}:")
        print(f"     Accuracy: {accuracy:.3f}")
        print(f"     AUC: {auc:.3f}")
        
        # Detailed classification report
        if name == "Logistic Regression":
            print(f"     Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['Human', 'Russian']))
    
    return lr_model, rf_model, scaler, feature_cols, X_test, y_test, y_train

def analyze_russian_patterns(lr_model, feature_cols, features_df):
    """Analyze which patterns most strongly indicate Russian speakers"""
    print("\n🔍 ENHANCED RUSSIAN PATTERN ANALYSIS")
    print("=" * 60)
    
    # Feature importance analysis
    coefficients = lr_model.coef_[0]
    feature_importance = list(zip(feature_cols, coefficients))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\n🏆 TOP RUSSIAN vs HUMAN LINGUISTIC INDICATORS:")
    print("(Positive = more Russian-like, Negative = more Human-like)")
    
    for i, (feature, coef) in enumerate(feature_importance[:20]):
        direction = "🇷🇺" if coef > 0 else "🇺🇸"
        print(f"   {i+1:2d}. {direction} {feature:30} = {coef:+.3f}")
    
    # Compare pattern frequencies between groups
    print(f"\n📊 PATTERN FREQUENCY COMPARISON (Russian vs Human):")
    
    russian_data = features_df[features_df['is_russian'] == 1]
    human_data = features_df[features_df['is_russian'] == 0]
    
    pattern_features = [col for col in feature_cols if 'count' in col or 'density' in col]
    
    for feature in pattern_features[:15]:
        russian_mean = russian_data[feature].mean()
        human_mean = human_data[feature].mean()
        difference = russian_mean - human_mean
        
        indicator = "🇷🇺" if difference > 0 else "🇺🇸"
        print(f"   {indicator} {feature:35} Russian: {russian_mean:.4f}, Human: {human_mean:.4f}, Diff: {difference:+.4f}")

def create_enhanced_visualizations(lr_model, rf_model, X_test, y_test, feature_cols, features_df):
    """Create enhanced visualizations"""
    print("\n📈 Creating enhanced linguistic analysis visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Feature Importance
    coefficients = lr_model.coef_[0]
    top_features_idx = np.argsort(np.abs(coefficients))[-15:]
    
    colors = ['red' if coefficients[i] > 0 else 'blue' for i in top_features_idx]
    axes[0,0].barh(range(len(top_features_idx)), coefficients[top_features_idx], color=colors)
    axes[0,0].set_yticks(range(len(top_features_idx)))
    axes[0,0].set_yticklabels([feature_cols[i] for i in top_features_idx], fontsize=8)
    axes[0,0].set_title('Top 15 Features: Russian vs Human')
    axes[0,0].set_xlabel('Coefficient (Red=Russian, Blue=Human)')
    
    # 2. Prediction Distribution
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    
    axes[0,1].hist(lr_probs[y_test == 0], bins=30, alpha=0.7, label='Human', color='blue')
    axes[0,1].hist(lr_probs[y_test == 1], bins=30, alpha=0.7, label='Russian Bot', color='red')
    axes[0,1].set_xlabel('Russian Pattern Probability')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('Probability Distribution')
    axes[0,1].legend()
    
    # 3. Confusion Matrix
    y_pred = lr_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,2], cmap='Blues',
                xticklabels=['Human', 'Russian'], yticklabels=['Human', 'Russian'])
    axes[0,2].set_title('Confusion Matrix')
    axes[0,2].set_xlabel('Predicted')
    axes[0,2].set_ylabel('Actual')
    
    # 4. Pattern Category Comparison
    russian_data = features_df[features_df['is_russian'] == 1]
    human_data = features_df[features_df['is_russian'] == 0]
    
    categories = ['article_errors_density', 'preposition_confusion_density', 
                 'false_friends_density', 'russian_calques_density', 'misspelling_density']
    
    russian_means = [russian_data[cat].mean() if cat in russian_data.columns else 0 for cat in categories]
    human_means = [human_data[cat].mean() if cat in human_data.columns else 0 for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1,0].bar(x - width/2, russian_means, width, label='Russian', color='red', alpha=0.7)
    axes[1,0].bar(x + width/2, human_means, width, label='Human', color='blue', alpha=0.7)
    axes[1,0].set_xlabel('Pattern Categories')
    axes[1,0].set_ylabel('Average Density')
    axes[1,0].set_title('Pattern Category Comparison')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels([cat.replace('_density', '').replace('_', ' ') for cat in categories], rotation=45)
    axes[1,0].legend()
    
    # 5. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, lr_probs)
    auc_score = roc_auc_score(y_test, lr_probs)
    
    axes[1,1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    axes[1,1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    axes[1,1].set_xlabel('False Positive Rate')
    axes[1,1].set_ylabel('True Positive Rate')
    axes[1,1].set_title('ROC Curve')
    axes[1,1].legend()
    
    # 6. Text Length Distribution
    axes[1,2].hist(russian_data['word_count'], bins=30, alpha=0.7, label='Russian', color='red')
    axes[1,2].hist(human_data['word_count'], bins=30, alpha=0.7, label='Human', color='blue')
    axes[1,2].set_xlabel('Word Count')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].set_title('Text Length Distribution')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig('enhanced_russian_linguistic_analysis.png', dpi=300, bbox_inches='tight')
    print("   📊 Saved enhanced visualization to 'enhanced_russian_linguistic_analysis.png'")

def demonstrate_enhanced_detection(lr_model, scaler, feature_cols):
    """Demonstrate enhanced detection on sample texts"""
    print("\n🎯 ENHANCED DETECTION DEMONSTRATION")
    print("=" * 50)
    
    extractor = EnhancedRussianLinguisticExtractor()
    
    sample_texts = [
        {
            'name': 'Native English Speaker',
            'text': "I'm really excited about the new movie coming out next week. The trailer looks fantastic and I can't wait to see it with my friends!"
        },
        {
            'name': 'Russian Speaker (Mild)',
            'text': "I am very interested in this new film. It looks quite good and I want to see it in the cinema when it will be available."
        },
        {
            'name': 'Russian Speaker (Strong)',
            'text': "This film is very actual topic now. I will be to watch it definately because depends from good reviews. In russia we also make such movies."
        },
        {
            'name': 'Russian Bot Pattern',
            'text': "American people must understand that this situation is very actual. We need to make sport and take decision about future of our country."
        }
    ]
    
    for sample in sample_texts:
        features = extractor.extract_enhanced_features(sample['text'])
        features_array = np.array([features.get(col, 0) for col in feature_cols]).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        russian_prob = lr_model.predict_proba(features_scaled)[0, 1]
        
        print(f"\n📝 {sample['name']}:")
        print(f"   Text: \"{sample['text']}\"")
        print(f"   Russian Pattern Score: {russian_prob:.1%}")
        
        # Show detected patterns
        detected = []
        for feature, value in features.items():
            if ('count' in feature or 'density' in feature) and value > 0:
                detected.append(f"{feature}({value:.2f})")
        
        if detected:
            print(f"   Detected patterns: {', '.join(detected[:4])}")

def main():
    """Enhanced main function"""
    print("🇷🇺 ENHANCED RUSSIAN LINGUISTIC DETECTOR")
    print("=" * 60)
    print("Comparing Russian bot tweets vs Human tweets to identify linguistic patterns")
    print()
    
    try:
        # Load comparison data
        df_combined = load_comparison_data(n_russian=25000, n_human=25000)
        
        # Extract enhanced features
        features_df = extract_enhanced_features_batch(df_combined)
        
        # Train enhanced models
        lr_model, rf_model, scaler, feature_cols, X_test, y_test, y_train = train_enhanced_detector(features_df)
        
        # Analyze patterns
        analyze_russian_patterns(lr_model, feature_cols, features_df)
        
        # Create visualizations
        create_enhanced_visualizations(lr_model, rf_model, X_test, y_test, feature_cols, features_df)
        
        # Demonstrate detection
        demonstrate_enhanced_detection(lr_model, scaler, feature_cols)
        
        # Save models
        joblib.dump(lr_model, MODELS_DIR / "enhanced_russian_detector_lr.pkl")
        joblib.dump(rf_model, MODELS_DIR / "enhanced_russian_detector_rf.pkl")
        joblib.dump(scaler, MODELS_DIR / "enhanced_russian_scaler.pkl")
        
        with open(MODELS_DIR / "enhanced_russian_features.txt", 'w') as f:
            for feature in feature_cols:
                f.write(f"{feature}\n")
        
        print(f"\n✅ SUCCESS! Enhanced Russian linguistic detector created")
        print(f"\n🎯 RESULTS SUMMARY:")
        russian_accuracy = accuracy_score(y_test, lr_model.predict(scaler.transform(X_test)))
        russian_auc = roc_auc_score(y_test, lr_model.predict_proba(scaler.transform(X_test))[:, 1])
        
        print(f"   • Compared {(y_train == 1).sum():,} Russian vs {(y_train == 0).sum():,} Human tweets")
        print(f"   • Final Accuracy: {russian_accuracy:.1%}")
        print(f"   • Final AUC: {russian_auc:.3f}")
        print(f"   • Can distinguish Russian linguistic patterns from native English")
        
        print(f"\n📁 Files created:")
        print(f"   • enhanced_russian_detector_lr.pkl")
        print(f"   • enhanced_russian_detector_rf.pkl")
        print(f"   • enhanced_russian_scaler.pkl")
        print(f"   • enhanced_russian_linguistic_analysis.png")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have both Russian troll tweets and legitimate tweets datasets.")

if __name__ == "__main__":
    main() 