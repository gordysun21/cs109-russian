# Artillery Triangulation System - CS 109 Extra Credit Project

A comprehensive implementation of sound-based artillery position estimation using Maximum Likelihood Estimation (MLE) and Monte Carlo simulation, designed to demonstrate key concepts from Stanford CS 109 (Introduction to Applied Statistics).

## 🎯 Project Overview

This project implements a sophisticated artillery triangulation system that uses acoustic sensors to estimate the position of artillery based on sound intensity measurements. The system demonstrates fundamental statistical concepts including parameter estimation, uncertainty quantification, and probabilistic modeling.

## 📁 Project Structure

```
CS109Project/
├── data/
│   └── russian_troll_tweets/     # Russian troll tweets dataset (656MB)
│       ├── IRAhandle_tweets_1.csv
│       ├── IRAhandle_tweets_2.csv
│       └── ... (9 files total)
├── scripts/
│   ├── download_dataset.py       # Kaggle dataset downloader
│   ├── copy_dataset.py          # Data organization utility
│   ├── data_utils.py            # Dataset loading utilities
│   ├── explore_data.py          # Data exploration
│   └── bot_detection_analysis.py # Bot detection modeling analysis
├── demo_triangulation.py        # Educational demonstration
├── interactive_triangulation_app.py    # Interactive GUI application
├── interactive_ukraine_triangulation_app.py  # Ukraine map version
├── artillery_triangulation.py   # Core triangulation system
├── run_project.py              # Project launcher
└── requirements.txt            # Python dependencies
```

## 🤖 Russian Troll Tweets Dataset Analysis

**NEW**: This project now includes analysis of the Russian Troll Tweets dataset from FiveThirtyEight/Kaggle for bot detection research.

### Dataset Overview
- **Source**: FiveThirtyEight Russian Troll Tweets (Kaggle)
- **Size**: 656MB across 9 CSV files
- **Records**: ~3.1M tweets from Russian troll accounts
- **Time Range**: 2012-2018
- **Purpose**: Understanding bot behavior patterns for detection models

### Key Limitation for Supervised Learning
⚠️ **Important**: This dataset contains **ONLY** Russian bot/troll tweets. For supervised machine learning (like logistic regression to predict bot probability), you need both:
- ✅ **Positive examples** (bot tweets) - This dataset provides these
- ❌ **Negative examples** (legitimate human tweets) - **NOT included**

### Data Access
Use the provided utilities to work with the dataset:

```python
from data_utils import load_single_file, load_sample, print_dataset_summary

# Quick overview
print_dataset_summary()

# Load first file for analysis
df = load_single_file(1)

# Load a small sample for testing
sample = load_sample(n_samples=1000)
```

## 🔬 CS 109 Concepts Demonstrated

### Original Artillery Triangulation System
- **Maximum Likelihood Estimation (MLE)**: Find the most likely artillery position given sensor measurements
- **Monte Carlo Simulation**: Quantify estimation uncertainty through repeated sampling
- **Probability Distributions**: Model measurement noise and estimation confidence
- **Bayesian Inference**: Update position estimates as new sensor data becomes available
- **Confidence Intervals**: Quantify the reliability of position estimates
- **Error Propagation**: Understand how measurement noise affects final estimates
- **Geometric Dilution of Precision**: Analyze how sensor placement affects accuracy

### Bot Detection Analysis
- **Single-Class Learning Problem**: Understanding limitations of imbalanced datasets
- **Feature Engineering**: Extracting meaningful characteristics from social media data
- **Unsupervised Learning**: Pattern detection without labeled negative examples
- **Model Validation Challenges**: Working with real-world data limitations

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd CS109Project
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Russian troll tweets dataset:**
   ```bash
   python download_dataset.py  # Downloads from Kaggle
   python copy_dataset.py      # Organizes into data/ folder
   ```

5. **For the Ukraine map version (optional):**
   ```bash
   pip install contextily geopandas folium xyzservices
   ```

### Running the Project

**Option 1: Use the launcher (recommended)**
```bash
python run_project.py
```

**Option 2: Run individual components**
```bash
# Educational demonstration
python demo_triangulation.py

# Interactive application
python interactive_triangulation_app.py

# Ukraine map version
python interactive_ukraine_triangulation_app.py

# Data exploration and bot analysis
python explore_data.py
python bot_detection_analysis.py
```

## 📦 Project Components

### 1. Artillery Triangulation System
The original CS 109 project demonstrating statistical concepts through artillery position estimation.

#### Educational Demo (`demo_triangulation.py`)
- Comprehensive step-by-step demonstration
- Statistical methodology explanations
- Progressive complexity building
- Educational visualizations

#### Interactive Applications
- Full-featured GUI with real-time parameter adjustment
- Live likelihood surface visualization
- Monte Carlo uncertainty analysis
- Ukraine map version with satellite imagery

### 2. Bot Detection Research (`NEW`)
Analysis of Russian troll tweets for understanding bot behavior patterns.

#### Data Utilities (`data_utils.py`)
Convenient functions for loading and working with the dataset:
- `load_single_file()`: Load individual CSV files
- `load_all_files()`: Combine all files (⚠️ Large - 3M+ tweets)
- `load_sample()`: Get random samples for testing
- `print_dataset_summary()`: Quick dataset overview

#### Exploration (`explore_data.py`)
Comprehensive analysis of the dataset structure and contents:
- Column descriptions and data types
- Sample content examination
- Temporal range analysis
- Account type categorization

#### Bot Detection Analysis (`bot_detection_analysis.py`)
Statistical modeling challenges and solutions:
- **Problem identification**: Single-class dataset limitations
- **Solution strategies**: Data combination approaches
- **Feature engineering**: Bot behavior pattern analysis
- **Alternative approaches**: Unsupervised and pseudo-labeling methods

## 🎯 Bot Detection Modeling: Key Insights

### The Challenge
This dataset is excellent for **understanding** bot behavior but **cannot directly train** a bot vs. human classifier because:

1. **Single-class problem**: Only bot tweets, no human comparison
2. **No ground truth**: Cannot calculate accuracy without human tweets
3. **Imbalanced learning**: Standard supervised methods won't work

### Recommended Solutions

#### 1. **Combine with Human Tweet Data**
```python
# Suggested approach
human_tweets = load_human_dataset()  # From Twitter API, Kaggle, etc.
bot_tweets = load_single_file(1)     # From this dataset

# Create balanced dataset
combined_data = create_balanced_dataset(human_tweets, bot_tweets)

# Train classifier
model = LogisticRegression()
model.fit(X_features, y_labels)  # Now you can predict probabilities!
```

#### 2. **Feature Engineering Focus**
Use this dataset to identify bot characteristics:
- Follower/following ratios
- Tweet frequency patterns
- Content characteristics (URLs, hashtags, mentions)
- Temporal posting patterns
- Account metadata analysis

#### 3. **Unsupervised Approaches**
- **One-Class SVM**: Treat bots as "normal", detect outliers
- **Isolation Forest**: Anomaly detection
- **Autoencoders**: Learn bot representations, use reconstruction error

#### 4. **Pseudo-Labeling**
Train on bot data, apply to unlabeled tweets, use confidence scores as probability estimates.

### Example Bot Characteristics Found
From our analysis of the dataset:
- **High retweet rate**: ~52% of tweets are retweets
- **URL usage**: ~40% of tweets contain URLs
- **Account categories**: RightTroll, LeftTroll, NonEnglish, etc.
- **Geographic targeting**: Primarily US-focused content
- **Temporal patterns**: Coordinated posting behaviors

## 🛠️ Dependencies

### Core Requirements
```
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
seaborn>=0.11.0
pandas>=1.3.0
kagglehub>=0.3.0
```

### Mapping Libraries (for Ukraine version)
```
contextily>=1.6.0
geopandas>=1.1.0
folium>=0.19.0
xyzservices>=2025.4.0
```

### Installation Notes
- **Data download**: Requires internet connection for Kaggle API
- **Virtual environment**: Recommended for dependency management
- **Platform support**: Windows, macOS, Linux

## 🎓 Educational Value

### For CS 109 Students
- **Hands-on experience** with MLE and Monte Carlo methods
- **Real-world data challenges** with the bot detection analysis
- **Visual understanding** of likelihood functions and confidence intervals
- **Practical application** of statistical concepts to both military and social media domains

### Key Learning Outcomes
1. **Parameter Estimation**: MLE implementation and interpretation
2. **Uncertainty Quantification**: Monte Carlo methods and confidence intervals
3. **Data Limitations**: Single-class learning problems and solutions
4. **Feature Engineering**: Extracting meaningful patterns from raw data
5. **Model Validation**: Challenges with imbalanced and incomplete datasets

## 🌍 Real-World Applications

### Artillery Triangulation
- Military intelligence and acoustic detection
- Emergency response and explosion localization
- Seismic monitoring and earthquake detection

### Bot Detection
- Social media platform security
- Election integrity and misinformation detection
- Academic research on information warfare
- Content moderation and platform safety

## 📈 Future Extensions

- Integration with real sensor hardware
- Real-time Twitter API integration for live bot detection
- Advanced NLP techniques for content analysis
- Graph analysis of bot network behaviors
- Temporal pattern analysis for coordinated campaigns

---

**Created for Stanford CS 109 Extra Credit Project**  
*Demonstrating the power of statistical methods in real-world applications*

**Dataset Attribution**: Russian Troll Tweets dataset provided by FiveThirtyEight, available on Kaggle # cs109-russian
