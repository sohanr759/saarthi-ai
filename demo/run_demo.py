#!/usr/bin/env python3
"""
Demo script for running Saarthi AI agent loop.

This script demonstrates how to:
1. Initialize the Speech-to-Text model
2. Create and run the AgentOrchestrator
3. Execute a complete agent loop for checking eligibility

Usage:
    python demo/run_demo.py

Or with audio file:
    python demo/run_demo.py --audio path/to/audio.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.agent.agent_loop import AgentOrchestrator
from backend.voice.stt import SpeechToText


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the demo."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def demo_without_audio() -> None:
    """Run a demo without audio input (simulated conversation)."""
    print("\n" + "=" * 70)
    print("Saarthi AI Demo - Without Audio Input")
    print("=" * 70 + "\n")

    # Initialize STT model (Hindi language)
    print("Initializing Speech-to-Text model...")
    stt = SpeechToText(target_language="hi", model_size="tiny")

    # Create orchestrator
    print("Creating AgentOrchestrator...")
    orchestrator = AgentOrchestrator(
        stt_model=stt,
        native_language="hi",
        max_retries=5,
    )

    # Run agent loop
    print("\nStarting agent loop for CHECK_ELIGIBILITY goal...")
    print("-" * 70)

    result = orchestrator.run(
        user_goal="CHECK_ELIGIBILITY",
        initial_intent="START",
        audio_input=None,  # No audio for this demo
    )

    print("-" * 70)
    print(f"\nFinal Result: {result['status']} (completed in {result['turns']} turns)")
    print("\n" + "=" * 70)


def demo_with_audio(audio_path: str) -> None:
    """Run a demo with audio file input."""
    print("\n" + "=" * 70)
    print("Saarthi AI Demo - With Audio Input")
    print("=" * 70 + "\n")

    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    # Initialize STT model (Hindi language)
    print(f"Initializing Speech-to-Text model...")
    stt = SpeechToText(target_language="hi", model_size="tiny")

    # Create orchestrator
    print("Creating AgentOrchestrator...")
    orchestrator = AgentOrchestrator(
        stt_model=stt,
        native_language="hi",
        max_retries=5,
    )

    # Run agent loop with audio
    print(f"\nProcessing audio file: {audio_path}")
    print("Starting agent loop for CHECK_ELIGIBILITY goal...")
    print("-" * 70)

    result = orchestrator.run(
        user_goal="CHECK_ELIGIBILITY",
        initial_intent="START",
        audio_input=str(audio_file),  # Pass audio file path
    )

    print("-" * 70)
    print(f"\nFinal Result: {result['status']} (completed in {result['turns']} turns)")
    print("\n" + "=" * 70)


def demo_interactive() -> None:
    """Run an interactive demo where user can provide multiple inputs."""
    print("\n" + "=" * 70)
    print("Saarthi AI Demo - Interactive Mode")
    print("=" * 70 + "\n")

    # Initialize STT model
    print("Initializing Speech-to-Text model...")
    stt = SpeechToText(target_language="hi", model_size="tiny")

    # Create orchestrator
    print("Creating AgentOrchestrator...")
    orchestrator = AgentOrchestrator(
        stt_model=stt,
        native_language="hi",
        max_retries=10,  # More retries for interactive mode
    )

    print("\nInteractive mode: Provide audio files or type 'quit' to exit")
    print("-" * 70)

    turn_count = 0
    while True:
        user_input = input("\nEnter audio file path (or 'quit' to exit): ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nExiting interactive mode.")
            break

        if not user_input:
            continue

        audio_file = Path(user_input)
        if not audio_file.exists():
            print(f"Error: File not found: {user_input}")
            continue

        turn_count += 1
        print(f"\n--- Turn {turn_count} ---")
        print(f"Processing: {user_input}")

        result = orchestrator.run(
            user_goal="CHECK_ELIGIBILITY",
            initial_intent="START",
            audio_input=str(audio_file),
        )

        print(f"Result: {result['status']} ({result['turns']} internal turns)")

    print("\n" + "=" * 70)


def main() -> None:
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(
        description="Run Saarthi AI agent loop demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo without audio (simulated)
  python demo/run_demo.py

  # Run demo with audio file
  python demo/run_demo.py --audio path/to/audio.wav

  # Run interactive demo
  python demo/run_demo.py --interactive

  # Verbose logging
  python demo/run_demo.py --verbose
        """,
    )

    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file for transcription",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (multiple audio inputs)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    try:
        if args.interactive:
            demo_interactive()
        elif args.audio:
            demo_with_audio(args.audio)
        else:
            demo_without_audio()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError running demo: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

