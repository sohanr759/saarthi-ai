# SYSTEM INSTRUCTIONS FOR CURSOR AI

You are building a production-grade, voice-first, agentic AI system.

Non-negotiable rules:
- Do NOT build a single-prompt chatbot
- Use explicit agent roles: Planner, Executor, Evaluator
- Maintain clear separation of concerns
- Internal reasoning must be in English
- ALL user-facing output must be in ONE native Indian language only
- Implement memory, tools, and failure handling explicitly
- Prefer deterministic logic over LLMs where possible

These rules apply to every module in this repository.