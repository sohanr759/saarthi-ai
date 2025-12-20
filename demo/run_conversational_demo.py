#!/usr/bin/env python3
"""
Conversational demo script for Saarthi AI.

This script simulates a live conversational agent where:
- User provides audio input (from files or microphone)
- Agent processes and responds
- Conversation continues turn-by-turn until completion

Usage:
    # With audio files (simulated conversation)
    python demo/run_conversational_demo.py --audio-files audio1.wav audio2.wav audio3.wav

    # Interactive mode (provide audio files one at a time)
    python demo/run_conversational_demo.py --interactive
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.agent.agent_loop import AgentOrchestrator
from backend.voice.stt import SpeechToText
from backend.voice.microphone import MicrophoneCapture


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the demo."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def conversational_demo_with_files(audio_files: List[str], user_goal: str = "CHECK_ELIGIBILITY") -> None:
    """
    Run a conversational demo using a sequence of audio files.

    This simulates a live conversation where each audio file represents
    one user turn in the conversation.
    """
    print("\n" + "=" * 70)
    print("Saarthi AI - Conversational Demo (File-based)")
    print("=" * 70 + "\n")

    # Initialize STT model
    print("Initializing Speech-to-Text model...")
    stt = SpeechToText(target_language="hi", model_size="tiny")

    # Create orchestrator
    print("Creating AgentOrchestrator...")
    orchestrator = AgentOrchestrator(
        stt_model=stt,
        native_language="hi",
        max_retries=20,  # Higher limit for multi-turn conversation
    )

    print(f"\nStarting conversational loop (goal: {user_goal})")
    print("Each audio file represents one user turn.\n")
    print("-" * 70)

    turn_number = 0
    conversation_active = True

    for audio_file_path in audio_files:
        audio_file = Path(audio_file_path)
        if not audio_file.exists():
            print(f"⚠️  Warning: Audio file not found: {audio_file_path}")
            print("Skipping this turn...\n")
            continue

        turn_number += 1
        print(f"\n[Turn {turn_number}]")
        print(f"📥 User audio: {audio_file_path}")

        # Process this conversational turn
        result = orchestrator.process_turn(
            user_goal=user_goal,
            audio_input=str(audio_file),
        )

        # Display agent response
        print(f"🤖 Agent response: {result['response_text']}")
        print(f"   Status: {result['status']}")
        print(f"   Plan action: {result['plan'].get('action', 'N/A')}")

        # Check if conversation should continue
        if result['status'] == "SUCCESS":
            print("\n✅ Conversation completed successfully!")
            conversation_active = False
            break
        elif result['status'] == "FAIL":
            print("\n❌ Conversation ended with failure.")
            conversation_active = False
            break
        elif not result['should_continue']:
            print("\n⚠️  Agent indicates conversation should end.")
            conversation_active = False
            break

        print("-" * 70)

    if conversation_active:
        print(f"\n📊 Conversation ended after {turn_number} turns.")
        print("Status: Incomplete (more user input may be needed)")

    print("\n" + "=" * 70)


def conversational_demo_interactive(user_goal: str = "CHECK_ELIGIBILITY") -> None:
    """
    Run an interactive conversational demo.

    User provides audio files one at a time, and the agent responds
    after each turn until the conversation completes.
    """
    print("\n" + "=" * 70)
    print("Saarthi AI - Interactive Conversational Demo")
    print("=" * 70 + "\n")

    # Initialize STT model
    print("Initializing Speech-to-Text model...")
    stt = SpeechToText(target_language="hi", model_size="tiny")

    # Create orchestrator
    print("Creating AgentOrchestrator...")
    orchestrator = AgentOrchestrator(
        stt_model=stt,
        native_language="hi",
        max_retries=20,
    )

    print(f"\nConversational mode active (goal: {user_goal})")
    print("Provide audio files one at a time. Type 'quit' to exit.\n")
    print("-" * 70)

    turn_number = 0

    while True:
        user_input = input(f"\n[Turn {turn_number + 1}] Enter audio file path (or 'quit'): ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n👋 Ending conversation.")
            break

        if not user_input:
            continue

        audio_file = Path(user_input)
        if not audio_file.exists():
            print(f"❌ Error: File not found: {user_input}")
            continue

        turn_number += 1
        print(f"\n📥 Processing: {audio_file.name}")

        # Process conversational turn
        result = orchestrator.process_turn(
            user_goal=user_goal,
            audio_input=str(audio_file),
        )

        # Display response
        print(f"\n🤖 Agent: {result['response_text']}")
        print(f"   Status: {result['status']} | Action: {result['plan'].get('action', 'N/A')}")

        # Check completion
        if result['status'] == "SUCCESS":
            print("\n✅ Conversation completed successfully!")
            break
        elif result['status'] == "FAIL":
            print("\n❌ Conversation ended with failure.")
            break
        elif not result['should_continue']:
            print("\n⚠️  Agent indicates conversation should end.")
            break

        print("-" * 70)

    print(f"\n📊 Total turns: {turn_number}")
    print("=" * 70)


def conversational_demo_microphone(user_goal: str = "CHECK_ELIGIBILITY", use_vad: bool = True) -> None:
    """
    Run a conversational demo with real-time microphone input.

    This mode:
    - Records audio directly from the microphone
    - Processes each recording turn-by-turn
    - Continues until conversation completes or user exits
    """
    print("\n" + "=" * 70)
    print("Saarthi AI - Real-Time Microphone Conversational Demo")
    print("=" * 70 + "\n")

    # Initialize STT model
    print("Initializing Speech-to-Text model...")
    stt = SpeechToText(target_language="hi", model_size="tiny")

    # Initialize microphone capture
    print("Initializing microphone capture...")
    try:
        mic = MicrophoneCapture(sample_rate=16000, channels=1)
        print("✅ Microphone ready!")
    except Exception as e:
        print(f"❌ Error initializing microphone: {e}")
        print("\nTroubleshooting:")
        print("1. Check microphone permissions")
        print("2. Ensure microphone is connected")
        print("3. Run: python -c 'from backend.voice.microphone import MicrophoneCapture; MicrophoneCapture.list_devices()'")
        sys.exit(1)

    # Create orchestrator
    print("Creating AgentOrchestrator...")
    orchestrator = AgentOrchestrator(
        stt_model=stt,
        native_language="hi",
        max_retries=20,
    )

    print(f"\n🎤 Real-time microphone mode active (goal: {user_goal})")
    print("=" * 70)
    print("Instructions:")
    print("  - Press ENTER to start recording")
    print("  - Speak into your microphone")
    print("  - Recording will stop automatically (VAD) or after max duration")
    print("  - Type 'quit' and press ENTER to exit")
    print("=" * 70 + "\n")

    turn_number = 0

    while True:
        user_input = input(f"\n[Turn {turn_number + 1}] Press ENTER to record (or 'quit' to exit): ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n👋 Ending conversation.")
            break

        # Record audio
        print("\n🎤 Recording... (speak now)")
        try:
            if use_vad:
                audio_data = mic.record_with_vad(max_duration=10.0, silence_threshold=0.01, silence_duration=1.5)
            else:
                print("   (Recording for 5 seconds, or press Ctrl+C to stop early)")
                audio_data = mic.record(duration=5.0, blocking=True)

            if len(audio_data) == 0:
                print("⚠️  No audio recorded. Please try again.")
                continue

            print(f"✅ Recorded {len(audio_data) / mic.sample_rate:.2f} seconds of audio")
            print("🔄 Processing...")

        except KeyboardInterrupt:
            print("\n⚠️  Recording interrupted.")
            continue
        except Exception as e:
            print(f"❌ Recording error: {e}")
            continue

        turn_number += 1

        # Process conversational turn
        try:
            result = orchestrator.process_turn(
                user_goal=user_goal,
                audio_input=audio_data,  # Pass numpy array directly
            )

            # Display response
            print(f"\n🤖 Agent: {result['response_text']}")
            print(f"   Status: {result['status']} | Action: {result['plan'].get('action', 'N/A')}")

            # Check completion
            if result['status'] == "SUCCESS":
                print("\n✅ Conversation completed successfully!")
                break
            elif result['status'] == "FAIL":
                print("\n❌ Conversation ended with failure.")
                break
            elif not result['should_continue']:
                print("\n⚠️  Agent indicates conversation should end.")
                break

        except Exception as e:
            print(f"❌ Error processing turn: {e}")
            continue

        print("-" * 70)

    print(f"\n📊 Total turns: {turn_number}")
    print("=" * 70)


def main() -> None:
    """Main entry point for the conversational demo."""
    parser = argparse.ArgumentParser(
        description="Run Saarthi AI conversational demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simulate conversation with multiple audio files
  python demo/run_conversational_demo.py --audio-files turn1.wav turn2.wav turn3.wav

  # Interactive mode (provide files one at a time)
  python demo/run_conversational_demo.py --interactive

  # Real-time microphone mode (speak directly into mic)
  python demo/run_conversational_demo.py --microphone

  # Microphone mode without VAD (fixed duration)
  python demo/run_conversational_demo.py --microphone --no-vad

  # Custom user goal
  python demo/run_conversational_demo.py --microphone --goal BROWSE_SCHEMES

  # Verbose logging
  python demo/run_conversational_demo.py --microphone --verbose
        """,
    )

    parser.add_argument(
        "--audio-files",
        nargs="+",
        help="List of audio files representing user turns in sequence",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (provide audio files one at a time)",
    )
    parser.add_argument(
        "--microphone",
        action="store_true",
        help="Run in real-time microphone mode (speak directly into mic)",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Voice Activity Detection (use fixed duration recording)",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="CHECK_ELIGIBILITY",
        help="User goal for the conversation (default: CHECK_ELIGIBILITY)",
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
        if args.microphone:
            conversational_demo_microphone(user_goal=args.goal, use_vad=not args.no_vad)
        elif args.interactive:
            conversational_demo_interactive(user_goal=args.goal)
        elif args.audio_files:
            conversational_demo_with_files(audio_files=args.audio_files, user_goal=args.goal)
        else:
            parser.print_help()
            print("\n❌ Error: One of --audio-files, --interactive, or --microphone must be specified.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

