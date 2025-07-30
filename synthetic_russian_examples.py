"""
Synthetic Russian Examples Generator

Creates high-probability Russian linguistic examples for testing the detector.
These examples incorporate multiple critical and moderate patterns.
"""

import json
from pathlib import Path

def generate_high_probability_russian_examples():
    """Generate synthetic examples with heavy Russian linguistic influence"""
    
    examples = {
        "critical_pattern_examples": [
            {
                "name": "Government Decision Making",
                "text": "This is very actual problem in our country. Government must take decision about economic situation. People depends from their choices.",
                "patterns": ["actual problem", "take decision", "depends from"]
            },
            {
                "name": "Sports Commentary", 
                "text": "russian athletes make sport very well in international competitions. They will be to participate in olympics next year.",
                "patterns": ["make sport", "will be to participate", "lowercase nationality"]
            },
            {
                "name": "Education Discussion",
                "text": "Students must make homework every day. This is actual topic for education system. Teachers put question about quality.",
                "patterns": ["make homework", "actual topic", "put question"]
            },
            {
                "name": "Political Analysis",
                "text": "american democracy depends from people participation. Different to other countries, they have actual problems with voting.",
                "patterns": ["depends from", "different to", "actual problems", "lowercase nationality"]
            },
            {
                "name": "Technology Opinion",
                "text": "I am knowing that internet has many informations about this actual topic. People are understanding importance of technology.",
                "patterns": ["am knowing", "actual topic", "are understanding", "many informations"]
            }
        ],
        
        "moderate_pattern_examples": [
            {
                "name": "Daily Life",
                "text": "In morning I go to university without breakfast. russian students have very good results in different subjects.",
                "patterns": ["in morning", "go to university", "lowercase nationality", "very good results"]
            },
            {
                "name": "Cultural Comparison",
                "text": "What kind of problems do american people have with healthcare? All this situation is quite bad thing for society.",
                "patterns": ["what kind of", "lowercase nationality", "all this", "quite bad thing"]
            },
            {
                "name": "Economic Discussion", 
                "text": "Economic crisis consist from many factors. People are afraid from unemployment and inflation in their countries.",
                "patterns": ["consist from", "afraid from", "many factors"]
            },
            {
                "name": "Academic Writing",
                "text": "Research has wrote about internet influence on young people. Many students go to school without proper preparation.",
                "patterns": ["has wrote", "internet", "go to school"]
            },
            {
                "name": "Social Media Commentary",
                "text": "In evening people use internet for communication. This is really good thing for modern society and international relations.",
                "patterns": ["in evening", "internet", "really good thing"]
            }
        ],
        
        "combined_heavy_examples": [
            {
                "name": "Political Editorial (Heavy Russian)",
                "text": "This is very actual situation in american politics today. Government must take decision about foreign policy. Many people depends from government choices, but different to other countries, they have democracy. I am knowing that internet has many informations about this topic. russian politicians make sport with international relations.",
                "patterns": ["Multiple critical and moderate patterns combined"]
            },
            {
                "name": "Educational Blog Post (Heavy Russian)",
                "text": "Students must make homework in evening after they go to university. This is actual problem for education system. Teachers put question: what kind of support do students need? Many young people are afraid from academic pressure. In morning they will be to study different subjects.",
                "patterns": ["Multiple critical and moderate patterns combined"]
            },
            {
                "name": "Sports Analysis (Heavy Russian)",
                "text": "russian athletes make sport very well in international competitions. This is actual topic for sports community. They depends from good training and proper equipment. Different to amateur athletes, professionals have very good results. What kind of preparation do they need for olympics?",
                "patterns": ["Multiple critical and moderate patterns combined"]
            },
            {
                "name": "Technology Review (Heavy Russian)",
                "text": "I am knowing that modern technology is actual problem for many people. internet has many informations about new devices. Users must take decision about which products to buy. In morning they go to work using different applications. This is really good thing for business development.",
                "patterns": ["Multiple critical and moderate patterns combined"]
            },
            {
                "name": "Social Commentary (Heavy Russian)",
                "text": "Very actual problem in modern society depends from social media influence. american young people are understanding importance of internet communication. Different to previous generation, they make homework using online resources. Teachers put question about this situation. All this changes consist from technological progress.",
                "patterns": ["Multiple critical and moderate patterns combined"]
            }
        ],
        
        "extreme_examples": [
            {
                "name": "Extreme Russian Patterns",
                "text": "this is very actual topic for discussion. russian people make sport activities in morning and evening. they depends from government decision about olympic participation. different to other countries, they will be to take decision about training programs. I am knowing that internet has many informations about this situation. students must make homework about sports achievements. what kind of problems do american athletes have? all this situation consist from political factors. people are afraid from international competition results.",
                "patterns": ["Maximum pattern density - almost every sentence has patterns"]
            },
            {
                "name": "Academic Paper Simulation",
                "text": "Research question: what kind of influence does internet have on education? this study depends from analysis of student behavior. different to previous studies, we are understanding that technology is actual problem. students make homework using online resources in evening. they go to university with different preparation levels. very important thing is that teachers put question about quality. russian education system has wrote many articles about this topic. all this information consist from various sources.",
                "patterns": ["Academic writing with heavy Russian influence"]
            }
        ]
    }
    
    return examples

def save_examples_as_json():
    """Save examples to JSON file for easy loading"""
    examples = generate_high_probability_russian_examples()
    
    with open("synthetic_russian_examples.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print("✅ Saved synthetic Russian examples to 'synthetic_russian_examples.json'")

def create_test_loader():
    """Create a simple script to load and test examples"""
    
    loader_code = '''"""
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
                print(f"\\n{i}. {example['name']}")
                print(f"   Text: \\"{example['text']}\\"")
                if 'patterns' in example:
                    print(f"   Patterns: {example['patterns']}")
                print()
        
        print("\\n🔍 HOW TO TEST:")
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
'''
    
    with open("test_synthetic_examples.py", "w") as f:
        f.write(loader_code)
    
    print("✅ Created 'test_synthetic_examples.py' for easy testing")

def main():
    """Generate synthetic Russian examples"""
    print("🎯 GENERATING SYNTHETIC RUSSIAN EXAMPLES")
    print("=" * 50)
    print()
    
    # Generate and display examples
    examples = generate_high_probability_russian_examples()
    
    print("📝 Generated example categories:")
    for category, example_list in examples.items():
        print(f"   • {category}: {len(example_list)} examples")
    
    total_examples = sum(len(example_list) for example_list in examples.values())
    print(f"   • Total examples: {total_examples}")
    print()
    
    # Show a few sample examples
    print("🎯 SAMPLE HIGH-PROBABILITY EXAMPLES:")
    print()
    
    # Show one from each category
    for category, example_list in examples.items():
        if example_list:
            example = example_list[0]
            print(f"📂 {category.replace('_', ' ').title()}:")
            print(f"   Name: {example['name']}")
            print(f"   Text: \"{example['text'][:100]}{'...' if len(example['text']) > 100 else ''}\"")
            print()
    
    # Save files
    save_examples_as_json()
    create_test_loader()
    
    print()
    print("🚀 READY TO TEST!")
    print("Run: python test_synthetic_examples.py")
    print("Or use the examples directly in the Russian linguistic GUI")

if __name__ == "__main__":
    main() 