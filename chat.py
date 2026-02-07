#!/usr/bin/env python3
#Installed PowerShell 7 and set the PowerShell to run locally installed Python and activate the venv automatically when running scripts from the project root. This allows the chat app to run with the correct dependencies without manual activation.
#I was trying to run it and I need to set it to current user.
"""
NASA RAG Chat with RAGAS Evaluation Integration
"""
import streamlit as st
import os
from typing import Dict, List, Optional

import rag_client as rag_client
import llm_client
import ragas_evaluator


st.set_page_config(
    page_title="NASA RAG Chat with Evaluation",
    page_icon="🚀",
    layout="wide"
)


def display_evaluation_metrics(scores: Dict[str, float]):
    if "error" in scores:
        samples = scores.get("samples")
        if samples:
            st.sidebar.info("Evaluation currently unavailable — try these sample questions:")
            for q in samples:
                st.sidebar.write(f"- {q}")
            return
        else:
            st.sidebar.error(f"Evaluation Error: {scores['error']}")
            return

    st.sidebar.subheader("📊 Response Quality")
    for metric_name, score in scores.items():
        if isinstance(score, (int, float)):
            st.sidebar.metric(
                label=metric_name.replace('_', ' ').title(),
                value=f"{score:.3f}"
            )
            st.sidebar.progress(min(max(float(score), 0.0), 1.0))


def main():
    st.title("🚀 NASA Space Mission Chat with Evaluation")
    st.markdown("Chat with AI about NASA space missions with real-time quality evaluation. I am new so I may make mistakes and not have all the answers yet, but I will do my best to help! Try asking about Apollo 11, Challenger, or any other NASA mission.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []

    with st.sidebar:
        st.header("🔧 Configuration")

        with st.spinner("Discovering ChromaDB backends..."):
            available_backends = rag_client.discover_chroma_backends()

        if not available_backends:
            st.error("No ChromaDB backends found!")
            st.info("Run embedding pipeline first:\n`python embedding_pipeline.py ingest --update-mode replace`")
            st.stop()

        backend_options = {k: v["display_name"] for k, v in available_backends.items()}
        selected_backend_key = st.selectbox(
            "Select Document Collection",
            options=list(backend_options.keys()),
            format_func=lambda x: backend_options[x]
        )
        selected_backend = available_backends[selected_backend_key]

        st.subheader("🔑 OpenAI Settings")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", "voc-2009866425126677464339669333e56cd9164.50366416")
        )
        if not openai_key:
            st.warning("Please enter your OpenAI API key")
            st.stop()

        # Make sure other modules can read these
        os.environ["OPENAI_API_KEY"] = "voc-2009866425126677464339669333e56cd9164.50366416"
        # Hard coded API Key os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

        model_choice = st.selectbox(
            "OpenAI Model",
            options=["gpt-5-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
        )

        st.subheader("🔍 Retrieval Settings")
        n_docs = st.slider("Documents to retrieve", 1, 10, 3)

        #Mission filter UI
        mission_filter = st.selectbox(
            "Mission Filter",
            ["All", "apollo11", "apollo13", "challenger"],
            help="Filter retrieval results by mission"
        )

        st.subheader("📊 Evaluation Settings")
        enable_evaluation = st.checkbox("Enable RAGAS Evaluation", value=True)

        if st.session_state.last_evaluation and enable_evaluation:
            display_evaluation_metrics(st.session_state.last_evaluation)

        with st.spinner("Initializing RAG system..."):
            collection = rag_client.initialize_rag_system(
                selected_backend["directory"],
                selected_backend["collection_name"]
            )

        if collection is None:
            st.error(
                "ChromaDB is not available in this environment. "
                "Install chromadb in the active Python environment or activate the project's venv."
            )
            st.stop()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about NASA space missions..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                mf = None if mission_filter == "All" else mission_filter

                docs_result = rag_client.retrieve_documents(
                    collection,
                    prompt,
                    n_docs,
                    mf
                )

                context = ""
                contexts_list = []
                if docs_result and docs_result.get("documents"):
                    context = rag_client.format_context(
                        docs_result["documents"][0],
                        docs_result["metadatas"][0]
                    )
                    contexts_list = docs_result["documents"][0]
                    st.session_state.last_contexts = contexts_list

                try:
                    response = llm_client.generate_response(
                        openai_key,
                        prompt,
                        context,
                        st.session_state.messages[:-1],
                        model_choice
                    )
                except Exception as e:
                    # Friendly fallback for LLM/runtime failures
                    response = "I'm sorry we can't answer this. Try a different question."
                    # Attempt to load sample questions to help the user
                    try:
                        import json, re
                        samples_path = os.path.join('.vs', 'test_questions.json')
                        if os.path.exists(samples_path):
                            raw = open(samples_path, 'r', encoding='utf-8').read()
                            # strip C-style comment blocks if present
                            raw_clean = re.sub(r'/\*.*?\*/', '', raw, flags=re.S).strip()
                            samples = json.loads(raw_clean)
                        else:
                            samples = []
                    except Exception:
                        samples = []

                st.markdown(response)

                # If we displayed the friendly fallback, show sample questions
                if response.startswith("I'm sorry we can't answer this"):
                    if samples:
                        st.info("Here are sample questions you can try:")
                        for s in samples:
                            q = s.get('question') if isinstance(s, dict) else str(s)
                            st.write(f"- {q}")
                    else:
                        st.info("Try rephrasing your question or ask about a different mission.")

                if enable_evaluation:
                    with st.spinner("Evaluating response quality..."):
                        st.session_state.last_evaluation = ragas_evaluator.evaluate_response_quality(
                            prompt,
                            response,
                            contexts_list
                        )

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()