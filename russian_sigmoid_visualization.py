"""
Russian Linguistic Pattern Detection - Sigmoid Visualization

This script visualizes how the Russian linguistic detection model uses
the sigmoid function to separate English text from Russian-influenced text
based on grammar patterns and linguistic features.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from pathlib import Path
import re
from collections import Counter
import seaborn as sns

class RussianSigmoidVisualizer:
    def __init__(self):
        self.models_dir = Path("models")
        self.load_models_and_create_samples()
        
    def load_models_and_create_samples(self):
        """Load Russian linguistic models and create text samples"""
        try:
            # Load Russian linguistic models
            self.russian_model = joblib.load(self.models_dir / "enhanced_russian_linguistic_lr.pkl")
            self.russian_scaler = joblib.load(self.models_dir / "enhanced_russian_linguistic_scaler.pkl")
            
            # Load feature names
            with open(self.models_dir / "enhanced_russian_features.txt", 'r') as f:
                self.feature_cols = [line.strip() for line in f.readlines()]
            
            print("✅ Russian linguistic models loaded successfully!")
            
        except Exception as e:
            print(f"⚠️ Error loading Russian models: {e}")
            print("Creating mock models for demonstration...")
            self.create_mock_models()
        
        # Create diverse text samples for analysis
        self.create_text_samples()
    
    def create_mock_models(self):
        """Create mock models for demonstration if real ones aren't available"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Create mock model with realistic coefficients
        self.russian_model = LogisticRegression()
        self.russian_model.coef_ = np.array([[-0.5, 2.8, 1.5, -0.8, 3.2, 0.7, -0.3, 1.9, 2.1, -0.4]])
        self.russian_model.intercept_ = np.array([-1.2])
        self.russian_model.classes_ = np.array([0, 1])
        
        self.russian_scaler = StandardScaler()
        self.russian_scaler.mean_ = np.zeros(10)
        self.russian_scaler.scale_ = np.ones(10)
        
        self.feature_cols = [
            'critical_russian_patterns', 'moderate_russian_patterns', 'weak_russian_patterns',
            'text_length', 'word_count', 'sentence_count', 'avg_word_length',
            'weighted_russian_score', 'rule_based_russian', 'grammar_errors'
        ]
        
        print("🔧 Using mock Russian linguistic model for demonstration")
    
    def create_text_samples(self):
        """Create diverse text samples ranging from perfect English to heavy Russian influence"""
        
        self.text_samples = [
            # Perfect English (0% Russian probability expected)
            {
                "text": "I'm excited about the upcoming election results. The candidates presented clear policies during the debates.",
                "category": "Perfect English",
                "expected_prob": 0.05,
                "patterns": []
            },
            {
                "text": "The weather forecast shows rain tomorrow. I need to remember to bring my umbrella to work.",
                "category": "Perfect English", 
                "expected_prob": 0.03,
                "patterns": []
            },
            
            # Minor Russian influence (10-30% expected)
            {
                "text": "The weather is very good today. I will go to university in the morning for my classes.",
                "category": "Minor Russian", 
                "expected_prob": 0.25,
                "patterns": ["go to university", "in morning"]
            },
            {
                "text": "Many people think this topic is actual for our society today.",
                "category": "Minor Russian",
                "expected_prob": 0.30,
                "patterns": ["actual topic"]
            },
            
            # Moderate Russian influence (40-70% expected)
            {
                "text": "Government must take decision about this actual problem. People depends from their choices in elections.",
                "category": "Moderate Russian",
                "expected_prob": 0.65,
                "patterns": ["take decision", "actual problem", "depends from"]
            },
            {
                "text": "russian athletes make sport very well. They train in morning and go to university for education.",
                "category": "Moderate Russian",
                "expected_prob": 0.70,
                "patterns": ["make sport", "in morning", "go to university", "lowercase nationality"]
            },
            
            # Heavy Russian influence (80-95% expected)
            {
                "text": "I am knowing this actual situation is very difficult. Government must take decision about depends from economy.",
                "category": "Heavy Russian",
                "expected_prob": 0.90,
                "patterns": ["am knowing", "actual situation", "take decision", "depends from"]
            },
            {
                "text": "In morning I go to university for make sport. This is actual problem for all russian students who depends from schedule.",
                "category": "Heavy Russian",
                "expected_prob": 0.95,
                "patterns": ["in morning", "go to university", "make sport", "actual problem", "depends from", "lowercase nationality"]
            },
            
            # Extreme Russian influence (95%+ expected)
            {
                "text": "I am knowing that actual problem is depends from government who must take decision about russian people who make sport in morning and go to university.",
                "category": "Extreme Russian",
                "expected_prob": 0.98,
                "patterns": ["am knowing", "actual problem", "depends from", "take decision", "make sport", "in morning", "go to university", "lowercase nationality"]
            }
        ]
        
        print(f"📝 Created {len(self.text_samples)} text samples for analysis")
    
    def extract_russian_features(self, text):
        """Extract Russian linguistic features from text"""
        if not text.strip():
            return np.zeros(len(self.feature_cols))
        
        text_lower = text.lower().strip()
        features = {}
        words = text_lower.split()
        sentences = re.split(r'[.!?]+', text)
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(words)
        features['sentence_count'] = max(1, len([s for s in sentences if s.strip()]))
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Critical Russian patterns (high confidence indicators)
        critical_patterns = [
            r'\b(actual|actually)\b(?=.*\b(topic|problem|situation|issue)\b)',  # actual problem/topic
            r'\b(make)\s+(sport|sports)\b',  # make sport
            r'\b(depends|rely|count)\s+(from|of)\b',  # depends from
            r'\b(take)\s+(decision|conclusion)\b',  # take decision
            r'\b(am|is|are)\s+(knowing|understanding)\b',  # am knowing
            r'\b(more|most)\s+(better|worse|important)\b',  # more better
            r'\b(go)\s+(by)\s+(foot|car|bus)\b',  # go by foot
        ]
        
        # Moderate Russian patterns (medium confidence)
        moderate_patterns = [
            r'\b(russian|american|ukrainian|british|german)\b',  # lowercase nationalities
            r'\bin\s+(morning|evening|afternoon)\b',  # in morning
            r'\bgo\s+to\s+(school|university|work|shop)\b',  # go to university
            r'\b(all|many|some)\s+this\b',  # all this
            r'\b(study)\s+(in|at)\s+(university|school)\b',  # study in university
            r'\b(very)\s+(good|bad|important|interesting)\b',  # very good (overuse)
        ]
        
        # Weak Russian patterns (lower confidence)
        weak_patterns = [
            r'\b(will)\s+(be)\s+(doing|going|coming)\b',  # will be doing (future continuous overuse)
            r'\b(should)\s+(to)\b',  # should to
            r'\b(must)\s+(to)\b',  # must to
            r'\b(can)\s+(to)\b',  # can to
            r'\b(enough)\s+(good|bad|big|small)\b',  # enough good
        ]
        
        # Count pattern occurrences
        critical_count = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in critical_patterns)
        moderate_count = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in moderate_patterns)
        weak_count = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in weak_patterns)
        
        features['critical_russian_patterns'] = critical_count
        features['moderate_russian_patterns'] = moderate_count
        features['weak_russian_patterns'] = weak_count
        
        # Weighted scoring (critical patterns count more)
        features['weighted_russian_score'] = critical_count * 3.0 + moderate_count * 1.5 + weak_count * 0.5
        
        # Rule-based classification
        features['rule_based_russian'] = 1 if (critical_count >= 1 or moderate_count >= 2 or weak_count >= 3) else 0
        
        # Grammar error estimation
        features['grammar_errors'] = critical_count + moderate_count * 0.5 + weak_count * 0.25
        
        # Fill remaining features with defaults
        for col in self.feature_cols:
            if col not in features:
                features[col] = 0
        
        # Convert to array in correct order
        feature_array = np.array([features.get(col, 0) for col in self.feature_cols])
        return feature_array
    
    def analyze_samples(self):
        """Analyze all text samples and compute predictions"""
        results = []
        
        for sample in self.text_samples:
            # Extract features
            features = self.extract_russian_features(sample['text'])
            
            # Scale features
            features_scaled = self.russian_scaler.transform(features.reshape(1, -1))
            
            # Get z-value (linear combination)
            z_value = self.russian_model.decision_function(features_scaled)[0]
            
            # Get probability
            probability = self.russian_model.predict_proba(features_scaled)[0, 1]
            
            results.append({
                'text': sample['text'],
                'category': sample['category'],
                'expected_prob': sample['expected_prob'],
                'patterns': sample['patterns'],
                'features': features,
                'z_value': z_value,
                'probability': probability,
                'classification': 'Russian' if probability > 0.5 else 'English'
            })
        
        return results
    
    def plot_russian_sigmoid_analysis(self):
        """Create comprehensive visualization of Russian linguistic detection"""
        # Analyze samples
        results = self.analyze_samples()
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Main sigmoid plot with sample points
        self.plot_main_russian_sigmoid(ax1, results)
        
        # 2. Pattern frequency analysis
        self.plot_pattern_analysis(ax2, results)
        
        # 3. Feature importance for Russian detection
        self.plot_russian_feature_importance(ax3)
        
        # 4. Probability vs expected comparison
        self.plot_probability_comparison(ax4, results)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed analysis
        self.print_sample_analysis(results)
    
    def plot_main_russian_sigmoid(self, ax, results):
        """Plot sigmoid function with Russian text samples"""
        # Extract z-values and probabilities
        z_values = [r['z_value'] for r in results]
        probabilities = [r['probability'] for r in results]
        categories = [r['category'] for r in results]
        
        # Create smooth sigmoid curve
        z_smooth = np.linspace(min(z_values) - 2, max(z_values) + 2, 200)
        sigmoid_smooth = 1 / (1 + np.exp(-z_smooth))
        
        # Plot sigmoid curve
        ax.plot(z_smooth, sigmoid_smooth, 'b-', linewidth=3, label='Sigmoid Function', alpha=0.8)
        
        # Color mapping for categories
        color_map = {
            'Perfect English': 'green',
            'Minor Russian': 'orange', 
            'Moderate Russian': 'red',
            'Heavy Russian': 'darkred',
            'Extreme Russian': 'maroon'
        }
        
        # Plot sample points
        for i, result in enumerate(results):
            color = color_map.get(result['category'], 'gray')
            marker = 'o' if result['classification'] == 'English' else '^'
            
            ax.scatter(result['z_value'], result['probability'], 
                      c=color, s=100, marker=marker, alpha=0.8,
                      edgecolors='black', linewidth=1,
                      label=result['category'] if i == 0 or result['category'] != results[i-1]['category'] else "")
        
        # Decision boundary
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, 
                  linewidth=2, label='Decision Boundary (P=0.5)')
        ax.axvline(x=0, color='orange', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Linear Combination (z)', fontsize=12)
        ax.set_ylabel('P(Russian)', fontsize=12)
        ax.set_title('Russian Linguistic Detection: Sigmoid Function', fontsize=14, fontweight='bold')
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        ax.text(min(z_values) + 0.5, 0.1, 'English\nRegion', fontsize=11, ha='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax.text(max(z_values) - 0.5, 0.9, 'Russian\nRegion', fontsize=11, ha='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    def plot_pattern_analysis(self, ax, results):
        """Plot pattern frequency analysis"""
        categories = [r['category'] for r in results]
        critical_patterns = [r['features'][0] for r in results]  # Assuming first feature is critical patterns
        moderate_patterns = [r['features'][1] for r in results]  # Assuming second feature is moderate patterns
        
        # Create grouped bar chart
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, critical_patterns, width, label='Critical Patterns', color='red', alpha=0.7)
        bars2 = ax.bar(x + width/2, moderate_patterns, width, label='Moderate Patterns', color='orange', alpha=0.7)
        
        ax.set_xlabel('Text Category', fontsize=12)
        ax.set_ylabel('Pattern Count', fontsize=12)
        ax.set_title('Russian Pattern Distribution by Category', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    def plot_russian_feature_importance(self, ax):
        """Plot feature importance for Russian detection"""
        coefficients = self.russian_model.coef_[0]
        
        # Get all features and their coefficients
        feature_names = self.feature_cols
        
        # Sort by absolute importance
        abs_coef = np.abs(coefficients)
        sorted_indices = np.argsort(abs_coef)
        
        # Take top features for display
        n_display = min(len(feature_names), 8)
        top_indices = sorted_indices[-n_display:]
        
        top_features = [feature_names[i] for i in top_indices]
        top_coef = coefficients[top_indices]
        
        # Clean feature names for display
        display_names = []
        for name in top_features:
            clean_name = name.replace('_', ' ').title()
            if len(clean_name) > 15:
                clean_name = clean_name[:12] + "..."
            display_names.append(clean_name)
        
        colors = ['red' if c > 0 else 'green' for c in top_coef]
        
        bars = ax.barh(range(len(top_coef)), top_coef, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_coef)))
        ax.set_yticklabels(display_names, fontsize=10)
        ax.set_xlabel('Coefficient Value', fontsize=12)
        ax.set_title('Feature Importance for Russian Detection\n(Red=Russian, Green=English)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_coef)):
            ax.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.2f}', 
                   va='center', ha='left' if val > 0 else 'right', fontsize=9)
    
    def plot_probability_comparison(self, ax, results):
        """Plot predicted vs expected probabilities"""
        categories = [r['category'] for r in results]
        predicted_probs = [r['probability'] for r in results]
        expected_probs = [r['expected_prob'] for r in results]
        
        # Create scatter plot
        colors = ['green', 'orange', 'red', 'darkred', 'maroon']
        unique_categories = list(dict.fromkeys(categories))  # Preserve order
        
        for i, category in enumerate(unique_categories):
            cat_indices = [j for j, c in enumerate(categories) if c == category]
            cat_predicted = [predicted_probs[j] for j in cat_indices]
            cat_expected = [expected_probs[j] for j in cat_indices]
            
            ax.scatter(cat_expected, cat_predicted, 
                      c=colors[i % len(colors)], s=100, alpha=0.7,
                      label=category, edgecolors='black')
        
        # Perfect prediction line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Prediction')
        
        # Decision boundary lines
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Expected P(Russian)', fontsize=12)
        ax.set_ylabel('Predicted P(Russian)', fontsize=12)
        ax.set_title('Predicted vs Expected Russian Probability', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def print_sample_analysis(self, results):
        """Print detailed analysis of each sample"""
        print("\n📊 DETAILED RUSSIAN LINGUISTIC ANALYSIS")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['category'].upper()}")
            print(f"   Text: \"{result['text']}\"")
            print(f"   🔍 Detected Patterns: {', '.join(result['patterns']) if result['patterns'] else 'None'}")
            print(f"   📏 Z-value: {result['z_value']:.3f}")
            print(f"   🎯 P(Russian): {result['probability']:.1%}")
            print(f"   📈 Expected: {result['expected_prob']:.1%}")
            print(f"   🏷️  Classification: {result['classification']}")
            
            # Show top features
            top_features = ['Critical Patterns', 'Moderate Patterns', 'Weak Patterns', 'Text Length', 'Word Count']
            feature_values = result['features'][:5]  # First 5 features
            print(f"   📊 Features: {', '.join([f'{name}={val:.1f}' for name, val in zip(top_features, feature_values)])}")

def main():
    """Run Russian linguistic sigmoid visualization"""
    print("🇷🇺 RUSSIAN LINGUISTIC PATTERN DETECTION - SIGMOID VISUALIZATION")
    print("=" * 70)
    print("🎯 This visualization shows how Russian grammar patterns and")
    print("   linguistic features are mapped through the sigmoid function.")
    print()
    
    # Create visualizer
    visualizer = RussianSigmoidVisualizer()
    
    # Run analysis and visualization
    print("🎨 Generating Russian linguistic analysis...")
    visualizer.plot_russian_sigmoid_analysis()
    
    print("\n✅ Russian linguistic visualization complete!")
    print()
    print("🔍 KEY INSIGHTS:")
    print("   • Perfect English texts cluster at low z-values (P < 0.5)")
    print("   • Russian-influenced texts move right as patterns increase")
    print("   • Critical patterns (like 'actual problem') have strong impact")
    print("   • Sigmoid smoothly transitions from English to Russian classification")
    print("   • Feature importance shows which patterns matter most")

if __name__ == "__main__":
    main() 