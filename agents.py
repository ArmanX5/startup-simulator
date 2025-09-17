# agents.py
from langchain_core.runnables import RunnableConfig
from datetime import datetime
import os
from llm_client import make_chat_model, ask_ai
from typing import Any, Dict

# --- Agent profiles (personalities) ---
DEFAULT_PROFILES = {
    "ceo": {
        "name": "Alex (CEO)",
        "tone": "decisive, strategic, product-focused",
        "instructions": "Focus on company vision, product-market fit, prioritization, and manager-style guidance."
    },
    "backend": {
        "name": "Sam (Backend)",
        "tone": "calm, precise, technical",
        "instructions": "Focus on implementation details, scalability, tradeoffs, and timelines."
    },
    "frontend": {
        "name": "Rina (Frontend)",
        "tone": "creative, user-focused, pragmatic",
        "instructions": "Focus on UX, UI tradeoffs, dev-estimates, and design choices."
    }
}

def init_state():
    return {
        "chat_log": [],  # list of dicts: {"agent":"ceo","text": "...", "ts": "2025-09-17T..."}
        "turn": 0,
        "agents": ["ceo", "backend", "frontend"],
        "profiles": DEFAULT_PROFILES,
    }

# Helper to append message
def append_message(state, agent_id, text):
    state["chat_log"].append({
        "agent": agent_id,
        "text": text,
        "ts": datetime.utcnow().isoformat() + "Z"
    })
    state["turn"] += 1
    return state

# Node: produce a message for a single agent
def agent_speak(
    state: Dict[str, Any],
    agent_id: str = "ceo",
    temperature: float = 0.6,
    model: str = "gpt-4o",
    **kwargs,
):
    """
    Ask the LLM to generate the next message for `agent_id` based on recent context.
    """
    chat_model = make_chat_model(temperature=temperature, model=model)

    # prepare prompt
    profile = state["profiles"][agent_id]
    recent = "\n".join([f"{state['profiles'][m['agent']]['name']}: {m['text']}" for m in state["chat_log"][-8:]])
    system_prompt = (
        f"You are {profile['name']}. Personality: {profile['tone']}. Instructions: {profile['instructions']}\n"
        "You are one participant in a small startup team chat. Keep messages short (1-3 sentences). "
        "If you request actions or ask questions, be explicit. Use no private info. Be realistic."
    )
    user_prompt = (
        f"Conversation so far:\n{recent}\n\n"
        f"As {profile['name']}, write the next chat message. Keep it concise and in-character."
    )

    ai_text = ask_ai(chat_model, system_prompt, user_prompt)
    state = append_message(state, agent_id, ai_text.strip())
    return state

# Orchestrator node: run one round (each agent speaks once in configured order)
def run_round(
    state: Dict[str, Any],
    temperature: float = 0.6,
    model: str = "gpt-4o",
    **kwargs,
):
    for agent in state["agents"]:
        state = agent_speak(state, agent_id=agent, temperature=temperature, model=model)
    return state
