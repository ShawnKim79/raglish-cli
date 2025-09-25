#!/usr/bin/env python3
"""
Main CLI entry point for Document RAG English Study application.
"""

import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from document_rag_english_study.cli import cli


def main() -> None:
    """Main entry point for the CLI application."""
    try:
        # Run the CLI directly using Click
        cli()
        
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()