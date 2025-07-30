"""
Sigmoid Function Visualization with Training Data

This script visualizes the sigmoid function using actual training data
from the bot detection system, showing how the model separates bots from humans.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from pathlib import Path
import seaborn as sns

class SigmoidTrainingVisualizer:
    def __init__(self):
        self.models_dir = Path("models")
        self.data_dir = Path("data")
        self.load_models_and_data()
        
    def load_models_and_data(self):
        """Load trained models and training data"""
        try:
            # Load models
            self.lr_model = joblib.load(self.models_dir / "logistic_regression_bot_detector.pkl")
            self.scaler = joblib.load(self.models_dir / "feature_scaler.pkl")
            
            # Load feature names
            with open(self.models_dir / "feature_columns.txt", 'r') as f:
                self.feature_cols = [line.strip() for line in f.readlines()]
            
            print("✅ Models loaded successfully!")
            
            # Try to load training data
            self.load_training_data()
            
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            print("Creating synthetic training data for demonstration...")
            self.create_synthetic_training_data()
    
    def load_training_data(self):
        """Load actual training data if available"""
        try:
            # Look for training data files
            possible_files = [
                "processed_training_data.csv",
                "bot_training_data.csv", 
                "X_train.csv",
                "training_features.csv"
            ]
            
            data_loaded = False
            for filename in possible_files:
                filepath = self.data_dir / filename
                if filepath.exists():
                    print(f"📊 Loading training data from {filename}")
                    self.training_data = pd.read_csv(filepath)
                    data_loaded = True
                    break
            
            if not data_loaded:
                # Try to find any CSV with training data
                csv_files = list(self.data_dir.glob("*.csv"))
                if csv_files:
                    print(f"📊 Using available data: {csv_files[0].name}")
                    self.training_data = pd.read_csv(csv_files[0])
                    data_loaded = True
            
            if not data_loaded:
                raise FileNotFoundError("No training data found")
                
            print(f"   Shape: {self.training_data.shape}")
            print(f"   Columns: {list(self.training_data.columns)[:10]}...")
            
        except Exception as e:
            print(f"⚠️ Could not load training data: {e}")
            self.create_synthetic_training_data()
    
    def create_synthetic_training_data(self):
        """Create synthetic training data that mimics real bot/human patterns"""
        print("🔧 Generating synthetic training data...")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Generate human accounts (label = 0)
        n_humans = 600
        humans = pd.DataFrame({
            'follower_following_ratio': np.random.gamma(2, 2, n_humans),  # Moderate ratios
            'tweets_per_day': np.random.gamma(1.5, 1.5, n_humans),  # Low activity
            'account_age_days': np.random.gamma(5, 200, n_humans),  # Older accounts
            'verified': np.random.choice([0, 1], n_humans, p=[0.95, 0.05]),  # Few verified
            'username_has_numbers': np.random.choice([0, 1], n_humans, p=[0.7, 0.3]),
            'profile_completeness': np.random.beta(5, 2, n_humans),  # Mostly complete
            'bot_label': [0] * n_humans
        })
        
        # Generate bot accounts (label = 1)
        n_bots = 400
        bots = pd.DataFrame({
            'follower_following_ratio': np.random.gamma(0.5, 0.1, n_bots),  # Very low ratios
            'tweets_per_day': np.random.gamma(3, 10, n_bots),  # High activity
            'account_age_days': np.random.gamma(1, 30, n_bots),  # Newer accounts
            'verified': [0] * n_bots,  # No verified bots
            'username_has_numbers': np.random.choice([0, 1], n_bots, p=[0.2, 0.8]),  # More numbers
            'profile_completeness': np.random.beta(2, 5, n_bots),  # Less complete
            'bot_label': [1] * n_bots
        })
        
        # Combine and shuffle
        self.training_data = pd.concat([humans, bots], ignore_index=True)
        self.training_data = self.training_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   Generated {len(self.training_data)} samples")
        print(f"   Humans: {len(humans)}, Bots: {len(bots)}")
    
    def compute_linear_combinations(self):
        """Compute linear combinations (z values) for training data"""
        # Prepare features to match model expectations
        feature_data = self.prepare_features()
        
        # Scale features
        X_scaled = self.scaler.transform(feature_data)
        
        # Compute linear combinations: z = X @ β
        # For logistic regression, decision_function gives us the linear combination
        z_values = self.lr_model.decision_function(X_scaled)
        
        return z_values
    
    def prepare_features(self):
        """Prepare features to match model expectations"""
        # Create feature matrix with same columns as training
        feature_matrix = np.zeros((len(self.training_data), len(self.feature_cols)))
        
        # Map available columns to feature matrix
        available_cols = self.training_data.columns
        
        for i, col in enumerate(self.feature_cols):
            if col in available_cols:
                feature_matrix[:, i] = self.training_data[col].fillna(0)
            elif 'follower_following_ratio' in col and 'follower_following_ratio' in available_cols:
                feature_matrix[:, i] = self.training_data['follower_following_ratio'].fillna(1.0)
            elif 'tweets_per_day' in col and 'tweets_per_day' in available_cols:
                feature_matrix[:, i] = self.training_data['tweets_per_day'].fillna(1.0)
            elif 'account_age' in col and 'account_age_days' in available_cols:
                feature_matrix[:, i] = self.training_data['account_age_days'].fillna(365)
            elif 'verified' in col and 'verified' in available_cols:
                feature_matrix[:, i] = self.training_data['verified'].fillna(0)
            else:
                # Set reasonable defaults based on column name
                if 'ratio' in col:
                    feature_matrix[:, i] = 1.0  # Neutral ratio
                elif 'age' in col:
                    feature_matrix[:, i] = 365  # 1 year default
                elif 'count' in col:
                    feature_matrix[:, i] = 100  # Moderate count
                else:
                    feature_matrix[:, i] = 0.0  # Zero default
        
        return feature_matrix
    
    def plot_sigmoid_with_data(self):
        """Create main visualization: sigmoid function with training data"""
        # Compute linear combinations
        z_values = self.compute_linear_combinations()
        
        # Get true labels
        if 'bot_label' in self.training_data.columns:
            y_true = self.training_data['bot_label']
        elif 'is_bot' in self.training_data.columns:
            y_true = self.training_data['is_bot']
        else:
            # Create labels based on patterns if not available
            y_true = (self.training_data['follower_following_ratio'] < 0.5).astype(int)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Main sigmoid plot with data points
        self.plot_main_sigmoid(ax1, z_values, y_true)
        
        # 2. Histogram of z-values by class
        self.plot_z_histogram(ax2, z_values, y_true)
        
        # 3. Feature importance (coefficients)
        self.plot_feature_importance(ax3)
        
        # 4. Probability distribution
        self.plot_probability_distribution(ax4, z_values, y_true)
        
        plt.tight_layout()
        plt.show()
    
    def plot_main_sigmoid(self, ax, z_values, y_true):
        """Plot sigmoid function with training data points"""
        # Create smooth sigmoid curve
        z_smooth = np.linspace(z_values.min() - 1, z_values.max() + 1, 200)
        sigmoid_smooth = 1 / (1 + np.exp(-z_smooth))
        
        # Plot sigmoid curve
        ax.plot(z_smooth, sigmoid_smooth, 'b-', linewidth=3, label='Sigmoid Function', alpha=0.8)
        
        # Plot training data points
        humans = y_true == 0
        bots = y_true == 1
        
        # Add jitter for better visibility
        jitter = np.random.normal(0, 0.02, len(z_values))
        
        ax.scatter(z_values[humans], y_true[humans] + jitter[humans], 
                  alpha=0.6, c='green', s=30, label=f'Humans ({humans.sum()})', 
                  marker='o', edgecolors='darkgreen')
        
        ax.scatter(z_values[bots], y_true[bots] + jitter[bots], 
                  alpha=0.6, c='red', s=30, label=f'Bots ({bots.sum()})', 
                  marker='^', edgecolors='darkred')
        
        # Decision boundary
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, 
                  linewidth=2, label='Decision Boundary (P=0.5)')
        ax.axvline(x=0, color='orange', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Linear Combination (z)', fontsize=12)
        ax.set_ylabel('P(Bot)', fontsize=12)
        ax.set_title('Sigmoid Function with Training Data', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        ax.text(z_values.min() + 0.5, 0.1, 'Human\nRegion', fontsize=11, ha='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax.text(z_values.max() - 0.5, 0.9, 'Bot\nRegion', fontsize=11, ha='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    def plot_z_histogram(self, ax, z_values, y_true):
        """Plot histogram of z-values by class"""
        humans = y_true == 0
        bots = y_true == 1
        
        ax.hist(z_values[humans], bins=30, alpha=0.7, color='green', 
               label=f'Humans ({humans.sum()})', density=True)
        ax.hist(z_values[bots], bins=30, alpha=0.7, color='red', 
               label=f'Bots ({bots.sum()})', density=True)
        
        ax.axvline(x=0, color='orange', linestyle='--', alpha=0.7, 
                  linewidth=2, label='Decision Boundary')
        
        ax.set_xlabel('Linear Combination (z)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Distribution of Linear Combinations', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_feature_importance(self, ax):
        """Plot feature importance (model coefficients)"""
        coefficients = self.lr_model.coef_[0]
        
        # Get top 10 most important features
        abs_coef = np.abs(coefficients)
        top_indices = np.argsort(abs_coef)[-10:]
        
        top_features = [self.feature_cols[i] for i in top_indices]
        top_coef = coefficients[top_indices]
        
        # Clean feature names for display
        display_names = []
        for name in top_features:
            if len(name) > 20:
                display_names.append(name[:17] + "...")
            else:
                display_names.append(name)
        
        colors = ['red' if c > 0 else 'green' for c in top_coef]
        
        bars = ax.barh(range(len(top_coef)), top_coef, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_coef)))
        ax.set_yticklabels(display_names, fontsize=10)
        ax.set_xlabel('Coefficient Value', fontsize=12)
        ax.set_title('Top 10 Feature Importance\n(Red=Bot, Green=Human)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_coef)):
            ax.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}', 
                   va='center', ha='left' if val > 0 else 'right', fontsize=9)
    
    def plot_probability_distribution(self, ax, z_values, y_true):
        """Plot predicted probability distribution"""
        probabilities = 1 / (1 + np.exp(-z_values))
        
        humans = y_true == 0
        bots = y_true == 1
        
        ax.hist(probabilities[humans], bins=30, alpha=0.7, color='green', 
               label=f'Humans ({humans.sum()})', density=True)
        ax.hist(probabilities[bots], bins=30, alpha=0.7, color='red', 
               label=f'Bots ({bots.sum()})', density=True)
        
        ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, 
                  linewidth=2, label='Decision Threshold')
        
        ax.set_xlabel('Predicted Probability P(Bot)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def print_model_analysis(self):
        """Print analysis of the model performance on training data"""
        z_values = self.compute_linear_combinations()
        probabilities = 1 / (1 + np.exp(-z_values))
        
        if 'bot_label' in self.training_data.columns:
            y_true = self.training_data['bot_label']
        elif 'is_bot' in self.training_data.columns:
            y_true = self.training_data['is_bot']
        else:
            y_true = (self.training_data['follower_following_ratio'] < 0.5).astype(int)
        
        predictions = (probabilities > 0.5).astype(int)
        accuracy = (predictions == y_true).mean()
        
        print("📊 MODEL ANALYSIS ON TRAINING DATA")
        print("=" * 50)
        print(f"📈 Training Accuracy: {accuracy:.1%}")
        print(f"🤖 Total Samples: {len(self.training_data)}")
        print(f"👥 Humans: {(y_true == 0).sum()} ({(y_true == 0).mean():.1%})")
        print(f"🤖 Bots: {(y_true == 1).sum()} ({(y_true == 1).mean():.1%})")
        print()
        
        print("📊 Z-VALUE STATISTICS:")
        print(f"   Range: [{z_values.min():.3f}, {z_values.max():.3f}]")
        print(f"   Mean: {z_values.mean():.3f}")
        print(f"   Std: {z_values.std():.3f}")
        print()
        
        print("🎯 PROBABILITY STATISTICS:")
        print(f"   Range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
        print(f"   Mean: {probabilities.mean():.3f}")
        print(f"   Human avg prob: {probabilities[y_true == 0].mean():.3f}")
        print(f"   Bot avg prob: {probabilities[y_true == 1].mean():.3f}")
        print()

def main():
    """Run the sigmoid visualization with training data"""
    print("📈 SIGMOID FUNCTION VISUALIZATION WITH TRAINING DATA")
    print("=" * 60)
    print("🎯 This visualization shows how logistic regression separates")
    print("   bots from humans using the sigmoid function.")
    print()
    
    # Create visualizer
    visualizer = SigmoidTrainingVisualizer()
    
    # Print model analysis
    visualizer.print_model_analysis()
    
    # Create visualization
    print("🎨 Generating visualization...")
    visualizer.plot_sigmoid_with_data()
    
    print("✅ Visualization complete!")
    print()
    print("🔍 KEY INSIGHTS:")
    print("   • Green points (humans) cluster on the left (low z-values)")
    print("   • Red points (bots) cluster on the right (high z-values)")
    print("   • Sigmoid function smoothly separates the two classes")
    print("   • Decision boundary at P(bot) = 0.5 (orange line)")
    print("   • Feature importance shows which features drive classification")

if __name__ == "__main__":
    main() 