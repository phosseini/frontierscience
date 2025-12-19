#!/usr/bin/env python3
"""
Test script to verify the FrontierScience benchmark installation.

This script tests core functionality without making API calls.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import litellm
        from dotenv import load_dotenv
        import tqdm
        print("✓ All dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Run: pip install -r requirements.txt")
        return False


def test_data_files():
    """Test that required data files exist."""
    print("\nTesting data files...")
    
    required_files = [
        "data/frontierscience_full.csv",
        "prompts/olympiad_judge_prompt.txt",
        "prompts/research_judge_prompt.txt"
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            all_exist = False
    
    return all_exist


def test_dataset_loading():
    """Test that dataset can be loaded."""
    print("\nTesting dataset loading...")
    try:
        from src.data_loader import FrontierScienceDataset
        
        dataset = FrontierScienceDataset("data/frontierscience_full.csv")
        stats = dataset.get_statistics()
        
        print(f"✓ Dataset loaded successfully")
        print(f"  - Total problems: {stats['total_problems']}")
        print(f"  - Olympiad: {stats['olympiad_problems']}")
        print(f"  - Research: {stats['research_problems']}")
        
        # Test getting problems
        olympiad = dataset.get_olympiad_problems(limit=1)
        research = dataset.get_research_problems(limit=1)
        
        if olympiad and research:
            print("✓ Can retrieve problems from both tracks")
            return True
        else:
            print("✗ Failed to retrieve problems")
            return False
            
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False


def test_prompt_loading():
    """Test that prompts can be loaded."""
    print("\nTesting prompt loading...")
    try:
        olympiad_prompt = Path("prompts/olympiad_judge_prompt.txt").read_text()
        research_prompt = Path("prompts/research_judge_prompt.txt").read_text()
        
        if "{problem}" in olympiad_prompt and "{problem}" in research_prompt:
            print("✓ Prompts loaded and contain expected placeholders")
            return True
        else:
            print("✗ Prompts missing expected placeholders")
            return False
    except Exception as e:
        print(f"✗ Prompt loading failed: {e}")
        return False


def test_env_setup():
    """Test environment variable setup."""
    print("\nTesting environment setup...")
    try:
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        
        if os.getenv('OPENAI_API_KEY'):
            print("✓ OPENAI_API_KEY is set")
            has_key = True
        else:
            print("⚠ OPENAI_API_KEY not set (required for evaluations)")
            has_key = False
        
        if Path('.env').exists():
            print("✓ .env file exists")
        else:
            print("⚠ .env file not found (copy .env.example to .env)")
        
        return has_key
    except Exception as e:
        print(f"✗ Environment setup test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 80)
    print("FrontierScience Benchmark Installation Test")
    print("=" * 80)
    print()
    
    tests = [
        ("Dependencies", test_imports),
        ("Data Files", test_data_files),
        ("Dataset Loading", test_dataset_loading),
        ("Prompt Loading", test_prompt_loading),
        ("Environment Setup", test_env_setup),
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
        print()
    
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")
    
    all_passed = all(results.values())
    
    print()
    if all_passed:
        print("🎉 All tests passed! The benchmark is ready to use.")
        print()
        print("Next steps:")
        print("  1. Set your API key in .env if not already done")
        print("  2. Run: python example_usage.py")
        print("  3. See QUICKSTART.md for evaluation examples")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print()
        print("Common fixes:")
        print("  - Run: pip install -r requirements.txt")
        print("  - Copy: cp .env.example .env")
        print("  - Add your OPENAI_API_KEY to .env")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())