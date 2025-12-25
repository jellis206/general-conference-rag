"""
Setup Verification Script

Checks that all required files and dependencies are in place
before running experiments.
"""

import os
import sys
from dotenv import load_dotenv


def check_env_file():
    """Check if .env file exists and has required variables."""
    print("Checking environment configuration...")

    if not os.path.exists(".env"):
        print("  ❌ .env file not found!")
        print("     Run: cp .env.example .env")
        print("     Then edit .env and add your OPENAI_API_KEY")
        return False

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("  ❌ OPENAI_API_KEY not set in .env file")
        print("     Edit .env and add your OpenAI API key")
        return False

    print("  ✓ Environment configuration OK")
    return True


def check_data_files():
    """Check if required data files exist."""
    print("\nChecking data files...")

    required_files = [
        ("free/free_talks.csv", "Free embeddings - talks"),
        ("free/free_paragraphs.csv", "Free embeddings - paragraphs"),
        ("free/free_3_clusters.csv", "Free embeddings - clusters"),
        ("openai/openai_talks.csv", "OpenAI embeddings - talks"),
        ("openai/openai_paragraphs.csv", "OpenAI embeddings - paragraphs"),
        ("openai/openai_3_clusters.csv", "OpenAI embeddings - clusters"),
    ]

    all_exist = True
    missing_files = []

    for filepath, description in required_files:
        if os.path.exists(filepath):
            print(f"  ✓ {description}")
        else:
            print(f"  ❌ {description} - {filepath} not found")
            all_exist = False
            missing_files.append(filepath)

    if not all_exist:
        print("\n  Missing data files! Run the pipeline:")
        print("    python scraper.py")
        print("    python free_embeddings.py")
        print("    python openai_embeddings.py")
        print("    python clusters.py")
        return False

    return True


def check_dependencies():
    """Check if required Python packages are installed."""
    print("\nChecking dependencies...")

    required_packages = [
        ("pandas", "Data processing"),
        ("numpy", "Numerical operations"),
        ("sentence_transformers", "Free embeddings"),
        ("openai", "OpenAI API"),
        ("sklearn", "Machine learning utilities"),
        ("dotenv", "Environment variables"),
    ]

    all_installed = True

    for package_name, description in required_packages:
        try:
            if package_name == "dotenv":
                __import__("dotenv")
            elif package_name == "sklearn":
                __import__("sklearn")
            else:
                __import__(package_name)
            print(f"  ✓ {package_name} - {description}")
        except ImportError:
            print(f"  ❌ {package_name} - {description} not installed")
            all_installed = False

    if not all_installed:
        print("\n  Missing dependencies! Run:")
        print("    pip install -r requirements.txt")
        return False

    return True


def main():
    """Run all verification checks."""
    print("="*80)
    print("General Conference RAG - Setup Verification")
    print("="*80)

    checks = [
        check_dependencies(),
        check_env_file(),
        check_data_files(),
    ]

    print("\n" + "="*80)

    if all(checks):
        print("✓ All checks passed! You're ready to run experiments.")
        print("="*80)
        print("\nNext steps:")
        print("  1. Run semantic search experiments:")
        print("     python semantic_search.py")
        print("\n  2. Run RAG experiments:")
        print("     python rag_query.py")
        print("\n  3. Run all experiments:")
        print("     python experiment_runner.py")
        print("\n  4. Test hybrid strategies:")
        print("     python hybrid_search.py")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
