# app.py
import streamlit as st
from agents import init_state, agent_speak, run_round
from llm_client import make_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

# -- App layout --
st.set_page_config(page_title="Startup Simulator", layout="wide")
st.title("🚀 Startup Multi-Agent Simulator (LangGraph + OpenAI)")

# session state for graph
if "state" not in st.session_state:
    st.session_state.state = init_state()
    st.session_state.running = False

cols = st.columns([2,1])
with cols[0]:
    st.subheader("Chat log")
    log_container = st.container()
    with log_container:
        for msg in st.session_state.state["chat_log"]:
            speaker = st.session_state.state["profiles"][msg["agent"]]["name"]
            st.markdown(f"**{speaker}**  \n{msg['text']}  \n<small>{msg['ts']}</small>", unsafe_allow_html=True)

with cols[1]:
    st.subheader("Agents")
    for aid, prof in st.session_state.state["profiles"].items():
        st.info(f"**{prof['name']}**\n\n{prof['instructions']}\n\nTone: {prof['tone']}", icon="🧠")

# Controls
st.sidebar.title("Controls")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.6, 0.05)
model = st.sidebar.text_input("Model name", value="gpt-4o")
if st.sidebar.button("Step: next agent"):
    # step 1 agent (use turn to determine who)
    agents = st.session_state.state["agents"]
    next_agent = agents[st.session_state.state["turn"] % len(agents)]
    st.session_state.state = agent_speak(st.session_state.state, agent_id=next_agent, temperature=temperature, model=model)
    # st.experimental_rerun()
    st.rerun()

if st.sidebar.button("Run: 1 full round"):
    st.session_state.state = run_round(st.session_state.state, temperature=temperature, model=model)
    # st.experimental_rerun()
    st.rerun()

n = st.sidebar.number_input("Run N rounds", min_value=1, max_value=100, value=3)
if st.sidebar.button("Run N rounds"):
    for _ in range(int(n)):
        st.session_state.state = run_round(st.session_state.state, temperature=temperature, model=model)
    # st.experimental_rerun()
    st.rerun()

if st.sidebar.button("Reset conversation"):
    st.session_state.state = init_state()
    # st.experimental_rerun()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("OpenAI key used from `OPENAI_API_KEY` env var.")
