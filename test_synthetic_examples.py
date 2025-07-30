"""
Test Loader for Synthetic Russian Examples

Run this to test the synthetic examples in the Russian linguistic detector.
"""

import json
import sys
from pathlib import Path

def load_and_test_examples():
    """Load synthetic examples and show how to test them"""
    
    try:
        with open("synthetic_russian_examples.json", "r") as f:
            examples = json.load(f)
        
        print("🇷🇺 SYNTHETIC RUSSIAN EXAMPLES")
        print("=" * 50)
        print()
        
        for category, example_list in examples.items():
            print(f"📂 {category.upper().replace('_', ' ')}")
            print("-" * 30)
            
            for i, example in enumerate(example_list, 1):
                print(f"\n{i}. {example['name']}")
                print(f"   Text: \"{example['text']}\"")
                if 'patterns' in example:
                    print(f"   Patterns: {example['patterns']}")
                print()
        
        print("\n🔍 HOW TO TEST:")
        print("1. Copy any text above")
        print("2. Run: python russian_linguistic_gui.py")  
        print("3. Paste the text in the analysis tab")
        print("4. Click 'Analyze Tweet'")
        print("5. Check the Russian probability score")
        print()
        print("Expected results: 90%+ Russian probability for most examples")
        
    except FileNotFoundError:
        print("❌ synthetic_russian_examples.json not found!")
        print("Run: python synthetic_russian_examples.py first")

if __name__ == "__main__":
    load_and_test_examples()
