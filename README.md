# 🗣️ Saarthi AI  
**Voice-First, Explainable Government Scheme Assistant (Native Language)**

Saarthi AI is a voice-driven, agentic AI system designed to help citizens discover and understand their eligibility for government welfare schemes through natural spoken conversations in a native Indian language.

Unlike conventional chatbots, Saarthi AI emphasizes explainability, robustness, and trust, making it suitable for real-world public service use cases.

---

## 🚩 Problem Statement

Accessing government welfare schemes is difficult for many citizens due to:
- Language barriers
- Complex eligibility criteria
- Lack of clarity on *why* they are eligible or not
- Low trust in automated systems

Most existing solutions provide static information or opaque chatbot responses, often leading to confusion and drop-offs.

---

## 💡 Solution Overview

Saarthi AI is a voice-first AI agent that:
- Interacts entirely in a single native Indian language
- Collects user information through conversational voice input
- Determines eligibility using deterministic, explainable logic
- Verbally explains eligibility decisions in simple language
- Handles errors, contradictions, and missing information gracefully

The system is built using an explicit agentic architecture rather than a single LLM prompt.

---

## ⭐ Key Differentiators (USP)

### 1. Explainable Eligibility Decisions
Instead of only stating outcomes, Saarthi AI clearly explains:
- Which eligibility criteria were satisfied
- Which criteria failed and why

This improves transparency and user trust, especially in public-sector contexts.

---

### 2. Explicit Agentic Workflow
The system follows a structured Planner → Executor → Evaluator loop:
- **Planner** decides the next action
- **Executor** performs tool calls and generates responses
- **Evaluator** validates outcomes and triggers retries if needed

This avoids brittle, single-prompt chatbot behavior.

---

### 3. Confidence-Aware Voice Interaction
- Speech-to-text confidence is monitored
- Contradictions across user inputs are detected
- Clarifications are requested only when necessary

This makes the system robust to real-world noise and accent variations.

---

### 4. Native-Language First Design
- All user interactions occur in one chosen Indian language
- No English fallback is used
- Internal reasoning is strictly separated from user-facing output

---

## 🧠 System Architecture

High-level flow:

User (Voice)
↓
Speech-to-Text (Language Locked)
↓
Planner Agent
↓
Executor Agent → Tools → Memory
↓
Evaluator Agent
↓
Text-to-Speech
↓
User (Voice)

---

## 🧩 Core Components

### Agent Modules
- **PlannerAgent** – Determines next steps based on intent and memory
- **ExecutorAgent** – Executes plans, calls tools, and generates responses
- **EvaluatorAgent** – Verifies completion and handles failures

### Memory
- **Conversation Memory** – Maintains dialogue history
- **User Profile Memory** – Stores structured user facts with confidence tracking

### Tools
- **Scheme Retriever** – Filters relevant government schemes
- **Eligibility Engine** – Rule-based, deterministic eligibility evaluation
- **Explainability Generator** – Converts reasoning traces into spoken explanations

### Voice Interface
- **Speech-to-Text (STT)** – Language-constrained transcription with confidence scores
- **Text-to-Speech (TTS)** – Native-language voice output

---

## 📂 Project Structure :

backend/

├── agent/
│ ├── planner.py
│ ├── executor.py
│ ├── evaluator.py
│ └── agent_loop.py
│
├── tools/
│ ├── scheme_retriever.py
│ ├── eligibility_engine.py
│ └── explainability.py
│
├── memory/
│ ├── conversation_memory.py
│ └── user_profile.py
│
├── voice/
│ ├── stt.py
│ ├── tts.py
│ └── microphone.py
│
├── data/
│ └── schemes.json
│
├── demo/
│ ├── demo_script.md
│ └── run_demo.py
│
├── architecture.md
├── SYSTEM_PROMPT.md
└── README.md

---

## 🚀 Quick Start

### Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd "/Users/sohanreddy76/Saarthi AI"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install system dependencies** (for audio processing):
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg` (or equivalent)
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

5. **Microphone permissions** (for real-time voice input):
   - macOS: Grant microphone access in System Preferences → Security & Privacy
   - Linux: Ensure your user is in the `audio` group
   - Windows: Grant microphone access in Privacy settings

   To test microphone access:
   ```bash
   python -c "from backend.voice.microphone import MicrophoneCapture; MicrophoneCapture.list_devices()"
   ```

### Running the Demo

The project includes a demo script that demonstrates the agent loop:

**Basic demo (without audio)**:
```bash
python demo/run_demo.py
```

**Demo with audio file**:
```bash
python demo/run_demo.py --audio path/to/your_audio.wav
```

**Interactive mode** (multiple audio inputs):
```bash
python demo/run_demo.py --interactive
```

**Verbose logging**:
```bash
python demo/run_demo.py --verbose
```

### Conversational Demo (Turn-by-Turn)

For a **live conversational experience** where the agent waits for user input between turns:

**Simulated conversation** (multiple audio files):
```bash
python demo/run_conversational_demo.py --audio-files turn1.wav turn2.wav turn3.wav
```

**Interactive conversational mode** (audio files):
```bash
python demo/run_conversational_demo.py --interactive
```

**Real-time microphone mode** (speak directly into mic):
```bash
python demo/run_conversational_demo.py --microphone
```

**Microphone mode without VAD** (fixed duration):
```bash
python demo/run_conversational_demo.py --microphone --no-vad
```

Conversational modes:
- Process one user audio input at a time
- Wait for your next input after each agent response
- Continue until the conversation goal is achieved
- Show agent responses and status after each turn

**Microphone mode features:**
- Real-time audio capture from your default microphone
- Voice Activity Detection (VAD) - automatically stops when you finish speaking
- Press ENTER to start recording, speak naturally
- Compatible with all conversation goals

### Using the Agent Programmatically

```python
from backend.voice.stt import SpeechToText
from backend.agent.agent_loop import AgentOrchestrator

# Initialize STT model
stt = SpeechToText(target_language="hi", model_size="tiny")

# Create orchestrator
orchestrator = AgentOrchestrator(
    stt_model=stt,
    native_language="hi",
    max_retries=5,
)

# Run agent loop (batch mode - runs to completion)
result = orchestrator.run(
    user_goal="CHECK_ELIGIBILITY",
    initial_intent="START",
    audio_input=None,  # or path to audio file
)

print(f"Status: {result['status']}, Turns: {result['turns']}")
```

**Conversational mode** (turn-by-turn):
```python
# Process one conversational turn at a time
turn_result = orchestrator.process_turn(
    user_goal="CHECK_ELIGIBILITY",
    audio_input="user_audio.wav",  # User's audio for this turn
)

print(f"Agent: {turn_result['response_text']}")
print(f"Status: {turn_result['status']}")
print(f"Continue: {turn_result['should_continue']}")

# Continue conversation if needed
if turn_result['should_continue']:
    next_turn = orchestrator.process_turn(
        user_goal="CHECK_ELIGIBILITY",
        audio_input="next_user_audio.wav",
    )
    # ... and so on
```

---

## 🧪 Demo Scenarios

The demo demonstrates three real-world cases:
1. Happy Path – User is eligible and receives a clear explanation
2. Contradiction Handling – Conflicting user inputs are detected and resolved
3. Graceful Failure – No eligible schemes, with polite reasoning

---

## 🔍 Design Philosophy

- Deterministic over probabilistic where correctness matters
- Explainability over cleverness
- Robustness over demo-only behavior
- Separation of concerns over monolithic prompts

---

## 🚀 Future Enhancements

- Vector-based scheme retrieval
- Multilingual support with language switching
- Integration with real government APIs
- Adaptive questioning based on user confidence

---

## 🏁 Conclusion

Saarthi AI demonstrates how agentic AI systems can be designed for high-trust, real-world applications, particularly in public service delivery. The project prioritizes clarity, reliability, and explainability—qualities essential for AI systems interacting with diverse populations.