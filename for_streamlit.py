from for_youtube_QA import (
    instantiate_model,
    extract_video_id,
    process_transcript,
    create_vector_store,
    create_retriever_chain,
    generate_response,
    LANGUAGE_MAPPING
)
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="YouTube Transcript Chatbot", layout="centered")
st.title("🎥 YouTube Transcript Chatbot")

# Initialize session state variables
if "model" not in st.session_state:
    st.session_state.model = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "split_transcript" not in st.session_state:
    st.session_state.split_transcript = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "source_language" not in st.session_state:
    st.session_state.source_language = None
if "target_language" not in st.session_state:
    st.session_state.target_language = None
if "video_url" not in st.session_state:
    st.session_state.video_url = ""

# Step 1: Enter YouTube URL
st.session_state.video_url = st.text_input(
    "📌 Please enter a YouTube video URL:", value=st.session_state.video_url
)

# Step 2: Select source language (store language code, e.g., 'hi')
if st.session_state.video_url:
    LANGUAGES = list(LANGUAGE_MAPPING.keys())
    selected_source = st.selectbox(
        "🌐 Select the video's language:",
        options=LANGUAGES,
        index=LANGUAGES.index("English") if "English" in LANGUAGES else 0,
    )
    st.session_state.source_language = LANGUAGE_MAPPING[selected_source]

# Step 3: Load & process transcript (pass video_id, source and target language codes)
if st.session_state.source_language:
    if st.button("📥 Load and Process Transcript"):
        if not api_key:
            st.error("API key not found in environment variables. Please check your .env file.")
        else:
            with st.spinner("Loading and processing transcript..."):
                try:
                    st.session_state.model = instantiate_model(api_key)
                    video_id = extract_video_id(st.session_state.video_url)
                    # Pass source and target language codes to process_transcript
                    # Use target_language if set, else fallback to source_language
                    target_lang = st.session_state.target_language or st.session_state.source_language
                    st.session_state.split_transcript = process_transcript(
                        video_id, st.session_state.source_language, target_lang
                    )
                    st.session_state.vector_store = create_vector_store(st.session_state.split_transcript)
                    st.success("Transcript loaded successfully!")
                except Exception as e:
                    st.error(f"Error during transcript loading: {e}")

# Step 4: Select target language (store language code, e.g., 'en')
if st.session_state.vector_store:
    LANGUAGES = list(LANGUAGE_MAPPING.keys())
    selected_target = st.selectbox(
        "🌐 Enter your preferred answer language:",
        options=LANGUAGES,
        index=LANGUAGES.index("English") if "English" in LANGUAGES else 0,
    )
    st.session_state.target_language = LANGUAGE_MAPPING[selected_target]

# Step 5: Prepare model & enable Q&A
if st.session_state.target_language:
    if st.button("✅ Prepare Model", key="prepare_model"):
        try:
            st.session_state.retriever = create_retriever_chain(
                st.session_state.vector_store,
                st.session_state.model,
                source_language=st.session_state.source_language,
                target_language=st.session_state.target_language
)

            st.success("Model prepared! Now you can ask questions.")
        except Exception as e:
            st.error(f"Failed to prepare model: {e}")

# Step 6: Ask questions
if st.session_state.retriever:
    question = st.text_input("❓ Enter your question about the video:", key="question_input")
    if st.button("🚀 Get Answer", key="get_answer"):
        if not question:
            st.warning("Please enter a question!")
        else:
            with st.spinner("Generating answer..."):
                try:
                    response = generate_response(
                        st.session_state.retriever,
                        question,
                        source_language=st.session_state,
                        target_language=st.session_state,
                    )
                    st.success("Answer:")
                    st.write(response)
                    st.session_state.chat_history.append({"question": question, "answer": response})
                except Exception as e:
                    st.error(f"Failed to generate response: {e}")

# Sidebar: Chat history
with st.sidebar:
    st.header("💬 Chat History (last 5)")
    if st.session_state.chat_history:
        preview_length = 50
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
            q_preview = (
                chat["question"][:preview_length] + "..." if len(chat["question"]) > preview_length else chat["question"]
            )
            a_preview = (
                chat["answer"][:preview_length] + "..." if len(chat["answer"]) > preview_length else chat["answer"]
            )
            st.markdown(f"**Q{i}:** {q_preview}")
            with st.expander(f"Answer {i} (click to expand)"):
                st.write(chat["answer"])
            st.markdown("---")
    else:
        st.write("No previous chats yet.")