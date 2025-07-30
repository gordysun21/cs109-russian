"""
Russian Linguistic Bot Detector GUI

A graphical user interface for detecting Russian speakers writing in English
based on common linguistic patterns and mistakes.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import re
from collections import Counter

# Setup paths
MODELS_DIR = Path("models")

class RussianLinguisticGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🇷🇺 Russian Linguistic Bot Detector")
        self.root.geometry("900x800")
        self.root.configure(bg='#f0f0f0')
        
        # Load models
        self.models_data = self.load_trained_models()
        
        # Create GUI elements
        self.create_widgets()
        
    def load_trained_models(self):
        """Load the trained Russian linguistic models"""
        try:
            # Load models
            lr_model = joblib.load(MODELS_DIR / "russian_linguistic_detector_lr.pkl")
            rf_model = joblib.load(MODELS_DIR / "russian_linguistic_detector_rf.pkl")
            scaler = joblib.load(MODELS_DIR / "russian_linguistic_scaler.pkl")
            
            # Load feature names
            with open(MODELS_DIR / "russian_linguistic_features.txt", 'r') as f:
                feature_cols = [line.strip() for line in f.readlines()]
            
            return lr_model, rf_model, scaler, feature_cols
            
        except FileNotFoundError as e:
            messagebox.showerror("Error", 
                f"Russian linguistic model files not found!\nPlease run the Russian linguistic detector training first.\nMissing: {e}")
            return None, None, None, None
    
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Title
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(pady=10)
        
        title_label = tk.Label(title_frame, text="🇷🇺 Russian Linguistic Bot Detector", 
                              font=('Arial', 18, 'bold'), bg='#f0f0f0', fg='#e74c3c')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="Detect Russian speakers writing in English based on linguistic patterns", 
                                 font=('Arial', 11), bg='#f0f0f0', fg='#7f8c8d')
        subtitle_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        
        # Analysis tab
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="🔍 Tweet Analysis")
        
        # Examples tab
        examples_frame = ttk.Frame(notebook)
        notebook.add(examples_frame, text="📋 Real Examples")
        
        # Patterns tab
        patterns_frame = ttk.Frame(notebook)
        notebook.add(patterns_frame, text="📚 Linguistic Patterns")
        
        notebook.pack(fill='both', expand=True)
        
        # Create widgets for each tab
        self.create_analysis_widgets(analysis_frame)
        self.create_examples_widgets(examples_frame)
        self.create_patterns_widgets(patterns_frame)
    
    def create_analysis_widgets(self, parent):
        """Create analysis tab widgets"""
        
        # Input section
        input_frame = ttk.LabelFrame(parent, text="📝 Tweet Text Input", padding=15)
        input_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(input_frame, text="Enter tweet text to analyze:", font=('Arial', 12, 'bold')).pack(anchor='w')
        
        self.tweet_text = scrolledtext.ScrolledText(input_frame, height=4, font=('Arial', 11), wrap=tk.WORD)
        self.tweet_text.pack(fill='x', pady=(5, 10))
        
        # Buttons
        button_frame = tk.Frame(input_frame, bg='white')
        button_frame.pack(fill='x')
        
        analyze_btn = tk.Button(button_frame, text="🔍 Analyze Tweet", 
                               font=('Arial', 12, 'bold'), bg='#e74c3c', fg='white',
                               command=self.analyze_tweet, pady=8)
        analyze_btn.pack(side='left', padx=5, fill='x', expand=True)
        
        clear_btn = tk.Button(button_frame, text="🗑️ Clear", 
                             font=('Arial', 12), bg='#95a5a6', fg='white',
                             command=self.clear_text, pady=8)
        clear_btn.pack(side='right', padx=5)
        
        # Results section
        results_frame = ttk.LabelFrame(parent, text="🎯 Analysis Results", padding=15)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=20, 
                                                     font=('Consolas', 10), wrap=tk.WORD)
        self.results_text.pack(fill='both', expand=True)
    
    def create_examples_widgets(self, parent):
        """Create examples tab with real Russian tweets"""
        
        # Load real examples
        real_examples = self.load_real_linguistic_examples()
        
        # Description
        desc_label = tk.Label(parent, text="Real Russian Troll Tweets (English) - Click to analyze:", 
                             font=('Arial', 12, 'bold'), bg='white')
        desc_label.pack(pady=10)
        
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg='white')
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add examples
        for name, data in real_examples.items():
            frame = ttk.LabelFrame(scrollable_frame, text=f"🇷🇺 {name}", padding=10)
            frame.pack(fill='x', padx=10, pady=5)
            
            # Show tweet content
            content_label = tk.Label(frame, text=f"Content: {data['content']}", 
                                   font=('Arial', 10), wraplength=800, justify='left')
            content_label.pack(anchor='w', pady=2)
            
            # Show metadata
            meta_label = tk.Label(frame, text=f"Author: {data['author']} | Type: {data['account_type']}", 
                                font=('Arial', 9), fg='#666')
            meta_label.pack(anchor='w', pady=2)
            
            # Analyze button
            analyze_btn = tk.Button(frame, text="📊 Analyze This Tweet", 
                                  command=lambda content=data['content']: self.load_example_tweet(content),
                                  bg='#e74c3c', fg='white', font=('Arial', 10))
            analyze_btn.pack(anchor='e', pady=(5, 0))
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", on_mousewheel)
    
    def create_patterns_widgets(self, parent):
        """Create patterns explanation tab"""
        
        # Title
        title_label = tk.Label(parent, text="🧠 Russian-English Linguistic Patterns", 
                              font=('Arial', 14, 'bold'), bg='white')
        title_label.pack(pady=10)
        
        # Create scrollable text
        patterns_text = scrolledtext.ScrolledText(parent, font=('Arial', 11), wrap=tk.WORD)
        patterns_text.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Add patterns explanation
        explanation = """📚 LINGUISTIC PATTERNS DETECTED:

🔸 ARTICLE USAGE (a, an, the):
   • Russian has no articles, so Russian speakers often:
   • Omit articles entirely: "I went to store" instead of "I went to the store"
   • Use wrong articles: "I saw a dog in the park" vs "I saw the dog in the park"
   • Overuse articles: "The democracy is important" instead of "Democracy is important"

🔸 PREPOSITION ERRORS:
   • Different preposition systems lead to mistakes:
   • "In Monday" instead of "On Monday"
   • "Depends from" instead of "Depends on"
   • "Different to" instead of "Different from"

🔸 WORD ORDER ISSUES:
   • Russian word order is more flexible, leading to:
   • "Yesterday I went to store" (placing time first)
   • "Very important is this issue" (emphasis patterns)
   • "About this we will talk later" (fronting objects)

🔸 VERB FORMS AND TENSE:
   • Aspect-based vs tense-based system differences:
   • Confusion with present perfect vs simple past
   • "I am knowing" instead of "I know" (progressive with stative verbs)
   • Missing auxiliary verbs in questions

🔸 COMMON TELLTALE PHRASES:
   • Direct translations from Russian:
   • "How to call" instead of "What is called"
   • "I am agree" instead of "I agree"
   • "More better" instead of "better"
   • "Different from me opinion" instead of "Different from my opinion"

🔸 PUNCTUATION AND STYLE:
   • Different punctuation rules:
   • Extra commas in complex sentences
   • Missing apostrophes in contractions
   • Different quotation mark usage

⚠️ IMPORTANT NOTES:
   • These patterns can also occur in other non-native speakers
   • Native speakers can make some of these mistakes too
   • Context and frequency matter more than individual errors
   • This detector looks for PATTERNS, not individual mistakes
"""
        
        patterns_text.insert(tk.END, explanation)
        patterns_text.config(state=tk.DISABLED)  # Make it read-only
    
    def load_real_linguistic_examples(self):
        """Load real Russian linguistic examples"""
        try:
            with open("russian_linguistic_examples.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            messagebox.showwarning("Examples Not Found", 
                "Real examples file not found. Run sample_real_accounts.py first.")
            return {}
        except Exception as e:
            messagebox.showerror("Error", f"Error loading examples: {e}")
            return {}
    
    def load_example_tweet(self, content):
        """Load example tweet into analysis tab"""
        self.tweet_text.delete(1.0, tk.END)
        self.tweet_text.insert(tk.END, content)
        
        # Switch to analysis tab
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Notebook):
                        child.select(0)  # Select first tab
                        break
        
        # Automatically analyze
        self.analyze_tweet()
    
    def clear_text(self):
        """Clear the tweet text"""
        self.tweet_text.delete(1.0, tk.END)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Text cleared. Ready for new analysis.\n")
    
    def extract_linguistic_features(self, text):
        """Extract enhanced linguistic features with sensitivity weighting"""
        if not text.strip():
            return None
        
        text_lower = text.lower().strip()
        features = {}
        words = text_lower.split()
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(words)
        features['sentence_count'] = max(1, len(re.findall(r'[.!?]+', text)))
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Enhanced Russian patterns with three tiers
        critical_patterns = [
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
        ]
        
        moderate_patterns = [
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
        ]
        
        weak_patterns = [
            r'[.!?]\s+[a-z]',                                                  # Lowercase after punctuation
            r'\s+,\s+',                                                        # Spaces before commas
            r'[.!?]{2,}',                                                      # Multiple punctuation
            r'\b(definately|recieve|seperate|occured|wether|wich|teh|thier)\b', # Common misspellings
            r'\b(loose)\b(?=.*\b(game|match|competition)\b)',                  # "loose" instead of "lose"
            r'\b(its)\b(?=.*\b(important|necessary|possible)\b)',              # "its" without apostrophe
        ]
        
        # Count patterns by tier
        critical_count = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in critical_patterns)
        moderate_count = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in moderate_patterns)
        weak_count = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) for pattern in weak_patterns)
        
        features['critical_russian_patterns'] = critical_count
        features['moderate_russian_patterns'] = moderate_count
        features['weak_russian_patterns'] = weak_count
        
        # Density features
        features['critical_pattern_density'] = critical_count / max(1, features['word_count'])
        features['moderate_pattern_density'] = moderate_count / max(1, features['word_count'])
        features['weak_pattern_density'] = weak_count / max(1, features['word_count'])
        
        # Weighted scoring
        features['weighted_russian_score'] = (
            critical_count * 3.0 +      # Critical patterns heavily weighted
            moderate_count * 1.5 +       # Moderate patterns moderately weighted  
            weak_count * 0.5            # Weak patterns lightly weighted
        )
        features['normalized_russian_score'] = features['weighted_russian_score'] / max(1, features['word_count'])
        
        # Rule-based classification (high sensitivity)
        features['rule_based_russian'] = 1 if (critical_count >= 1 or moderate_count >= 2 or weak_count >= 3) else 0
        
        # Article analysis
        articles = len(re.findall(r'\b(a|an|the)\b', text_lower))
        features['article_density'] = articles / max(1, features['word_count'])
        features['missing_articles'] = len(re.findall(r'\b(went\s+to|at|in|on)\s+[aeiou]', text_lower))
        
        # Grammar and style
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
        
        # Punctuation analysis
        features['exclamation_density'] = text.count('!') / max(1, features['text_length'])
        features['question_density'] = text.count('?') / max(1, features['text_length'])
        features['capitalization_errors'] = len(re.findall(r'[.!?]\s+[a-z]', text))
        
        return features
    
    def analyze_tweet(self):
        """Analyze the tweet text for Russian linguistic patterns"""
        if self.models_data[0] is None:
            messagebox.showerror("Error", "Models not loaded properly!")
            return
        
        text = self.tweet_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("No Text", "Please enter some text to analyze!")
            return
        
        try:
            # Show analyzing message
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "🔍 Analyzing linguistic patterns... Please wait.\n\n")
            self.root.update()
            
            # Extract features
            features = self.extract_linguistic_features(text)
            if features is None:
                self.results_text.insert(tk.END, "❌ Could not extract features from text.\n")
                return
            
            # Make predictions if models are available
            predictions = self.predict_russian_patterns(features)
            
            # Display results
            self.display_linguistic_results(text, features, predictions)
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error during analysis: {e}")
    
    def predict_russian_patterns(self, features):
        """Predict if text shows Russian linguistic patterns"""
        lr_model, rf_model, scaler, feature_cols = self.models_data
        
        if lr_model is None:
            return None
        
        # Create feature vector
        feature_vector = []
        for col in feature_cols:
            feature_vector.append(features.get(col, 0))
        
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Make predictions
        lr_prob = lr_model.predict_proba(X_scaled)[0, 1]
        rf_prob = rf_model.predict_proba(X_scaled)[0, 1]
        
        return {
            'logistic_regression': lr_prob,
            'random_forest': rf_prob,
            'average': (lr_prob + rf_prob) / 2
        }
    
    def display_linguistic_results(self, text, features, predictions):
        """Display the linguistic analysis results"""
        self.results_text.delete(1.0, tk.END)
        
        # Header
        self.results_text.insert(tk.END, "🇷🇺 ENHANCED RUSSIAN LINGUISTIC PATTERN ANALYSIS\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        
        # Text summary
        self.results_text.insert(tk.END, "📝 TEXT ANALYZED:\n")
        self.results_text.insert(tk.END, f'"{text[:200]}{"..." if len(text) > 200 else ""}"\n\n')
        
        # Enhanced pattern analysis
        self.results_text.insert(tk.END, "🎯 ENHANCED PATTERN DETECTION:\n")
        self.results_text.insert(tk.END, f"   🚨 Critical patterns: {features['critical_russian_patterns']} (high confidence indicators)\n")
        self.results_text.insert(tk.END, f"   ⚠️  Moderate patterns: {features['moderate_russian_patterns']} (moderate confidence indicators)\n")
        self.results_text.insert(tk.END, f"   💡 Weak patterns: {features['weak_russian_patterns']} (weak indicators)\n")
        self.results_text.insert(tk.END, f"   📊 Weighted score: {features['weighted_russian_score']:.2f}\n")
        self.results_text.insert(tk.END, f"   📈 Normalized score: {features['normalized_russian_score']:.3f}\n")
        self.results_text.insert(tk.END, f"   🤖 Rule-based classification: {'RUSSIAN' if features['rule_based_russian'] else 'ENGLISH'}\n\n")
        
        # High-sensitivity rule explanation
        self.results_text.insert(tk.END, "🔍 HIGH-SENSITIVITY RULES:\n")
        self.results_text.insert(tk.END, "   • 1+ critical patterns = RUSSIAN\n")
        self.results_text.insert(tk.END, "   • 2+ moderate patterns = RUSSIAN\n")
        self.results_text.insert(tk.END, "   • 3+ weak patterns = RUSSIAN\n\n")
        
        # Linguistic features
        self.results_text.insert(tk.END, "📊 LINGUISTIC FEATURES:\n")
        self.results_text.insert(tk.END, f"   • Word count: {features['word_count']}\n")
        self.results_text.insert(tk.END, f"   • Sentence count: {features['sentence_count']}\n")
        self.results_text.insert(tk.END, f"   • Average word length: {features['avg_word_length']:.2f}\n")
        self.results_text.insert(tk.END, f"   • Article density: {features['article_density']:.3f}\n")
        self.results_text.insert(tk.END, f"   • Missing articles: {features['missing_articles']}\n")
        self.results_text.insert(tk.END, f"   • Complex sentences: {features['complex_sentences']}\n")
        self.results_text.insert(tk.END, f"   • Comma usage: {features['comma_usage']:.3f}\n")
        self.results_text.insert(tk.END, f"   • Word repetition: {features['word_repetition']:.3f}\n")
        self.results_text.insert(tk.END, f"   • Capitalization errors: {features['capitalization_errors']}\n\n")
        
        # Predictions (if available)
        if predictions:
            self.results_text.insert(tk.END, "🎯 MACHINE LEARNING PREDICTIONS:\n")
            self.results_text.insert(tk.END, f"   📈 Logistic Regression: {predictions['logistic_regression']:.1%}\n")
            self.results_text.insert(tk.END, f"   🌲 Random Forest: {predictions['random_forest']:.1%}\n")
            self.results_text.insert(tk.END, f"   🎯 Average probability: {predictions['average']:.1%}\n\n")
            
            # Enhanced interpretation with high sensitivity
            avg_prob = predictions['average']
            self.results_text.insert(tk.END, "💡 ENHANCED INTERPRETATION:\n")
            if avg_prob > 0.7 or features['rule_based_russian']:
                self.results_text.insert(tk.END, "   🚨 HIGH probability of Russian linguistic patterns\n")
                self.results_text.insert(tk.END, "   Classification: LIKELY RUSSIAN SPEAKER\n")
            elif avg_prob > 0.3:
                self.results_text.insert(tk.END, "   ⚠️ MODERATE probability of Russian patterns\n")
                self.results_text.insert(tk.END, "   Classification: POSSIBLE RUSSIAN SPEAKER\n")
            else:
                self.results_text.insert(tk.END, "   ✅ LOW probability of Russian patterns\n")
                self.results_text.insert(tk.END, "   Classification: LIKELY NATIVE ENGLISH SPEAKER\n")
        else:
            self.results_text.insert(tk.END, "⚠️ Machine learning models not available. Using rule-based analysis only.\n")

def main():
    """Main function to run the Russian linguistic GUI"""
    root = tk.Tk()
    app = RussianLinguisticGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 