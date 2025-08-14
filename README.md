# YouTube RAG-Based Multilingual Q&A System

This project is a **Retrieval-Augmented Generation (RAG)** powered **Multilingual Q&A system** for YouTube videos, built with **LangChain** and **Groq LLM**.  
It allows users to:  

- **Extract** video transcripts directly from YouTube.  
- **Process & store** them in a vector database for semantic search.  
- **Ask questions** in their preferred language about the video’s content.  
- **Receive AI-generated answers** that are accurate and context-aware.  

The system supports **multiple languages** (via language mapping), uses static and dynamic promt and uses **LangChain’s retrieval chains** for efficient document search, combined with **Groq’s LLM** for fast and high-quality response generation.  

This is ideal for summarizing, understanding, or querying long YouTube videos without manually watching them.

## 📑 Table of Contents

1. [🚀 Project Overview](#project-overview)
2. [🚀 Tools, Libraries, and Packages](#tools-libraries-and-packages)
3. [🔑 GROQ API Key](#groq-api-key)
4. [📖 User Guide](#user-guide)
5. [🎯 Test Case 02: Bengali ➡️ English](#test-case-02-bengali--english)
6. [🎯 Test Case 03: English ➡️ Spanish](#test-case-03-english--spanish)
7. [🏆 Critical Challenges Faced & Solutions](#critical-challenges-faced--solutions)



## 🚀 Project Overview
- **`for_youtube_QA.py`** 🎯 : This file extracts the video ID, fetches and translates the transcript into chunks, builds a FAISS vector store, and creates a retriever chain for efficient search. Finally, it initializes the Groq LLM and enables users to query the video content via a CLI workflow.

The diagram below shows the end-to-end workflow of the `for_youtube_QA.py` script:

![Workflow of the for_youtube_QA](/QA.png)

- **`for_streamlit.py`** 🎯 This outlines session state initialization, sidebar chat history management, and the step-by-step process for entering a YouTube URL, selecting languages, processing transcripts, preparing the model, and enabling Q&A.

This diagram below illustrates the workflow of `for_streamlit.py`, starting from environment setup and variable loading to Streamlit page configuration.

![Workflow of the for_streamlit.py](/ST.png)

## 🚀 Tools, Libraries, and Packages

This project leverages a mix of powerful Python libraries, cutting-edge LLMs, and smart utilities to turn YouTube videos into an interactive Q&A experience.  

---

### 🐍 Python Libraries
| Library | Purpose |
|---------|---------|
| `os` | Manage environment variables like API keys. |
| `dotenv` | Load `.env` files securely. |
| `re` | Extract YouTube video IDs with regex magic. |
| `youtube_transcript_api` | Fetch transcripts from YouTube videos effortlessly. |
| `deep_translator` | Translate text between multiple languages using Google Translate. |
| `langchain` | Build advanced LLM-based pipelines and retrieval chains. |
| `langchain_community.vectorstores` (`FAISS`) | Store and search embeddings quickly with FAISS. |
| `langchain_community.embeddings` (`HuggingFaceEmbeddings`) | Generate high-quality multilingual text embeddings. |
| `langchain.text_splitter` (`RecursiveCharacterTextSplitter`) | Split long transcripts into manageable chunks. |
| `langchain.schema` (`Document`) | Represent text chunks as structured documents. |
| `langchain.prompts` (`ChatPromptTemplate`) | Create dynamic prompts for LLM input. |
| `langchain.chains.combine_documents` (`create_stuff_documents_chain`) | Merge multiple document chunks into a single context for the LLM. |
| `langchain.chains` (`create_retrieval_chain`) | Build retrieval-based Q&A chains over your documents. |

---

### 🤖 LLM & Embeddings
| Model / Class | Purpose |
|---------------|---------|
| `ChatGroq` | Groq LLM for answering questions intelligently. |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Multilingual embedding model from HuggingFace for generating vector representations. |

---

### 🛠 Other Cool Tools
| Tool / Concept | Purpose |
|----------------|---------|
| **FAISS** | Lightning-fast vector database for similarity search. |
| **YouTube Transcript API** | Programmatically fetch video transcripts. |
| **Text Splitters** | Handle large transcripts by splitting into smaller chunks for the LLM. |
| **Environment Variables** | Store sensitive credentials like `GROQ_API_KEY` securely. |
| **Google Translator** | Auto-translate transcripts and answers across multiple languages. |

---

> 💡 This setup makes it possible to **ask questions about any YouTube video** in your preferred language and get accurate, concise answers powered by LLMs.


## 🔑 GROQ API Key

To unlock the power of **Groq LLM** in this project, you'll need your **GROQ API Key**.  
This key allows the app to connect to the Groq API and generate intelligent responses from YouTube transcripts. Put the key in .env file, here set the name as GROQ_API_KEY otherwise it won't work. **Use your own API key**.

> **Example**:

```python
GROQ_API_KEY = "Use Your Own API Key"
```



## 📖 User Guide

- **📝 Step 01:** Copy the URL of the YouTube video you want to process.

![step1](/Snapshots/Step1.png)

- **📝 Step - 02:** Scroll down the video and click on “Show Transcript” to view the video’s transcript.

![step2](/Snapshots/step2.png)

In this video, the transcript language is Hindi.
![step2](/Snapshots/step2_2.png)

- **📝 Step - 03:** Open the streamlit interface **YouTube Transcript Chatbot**, paste the copied URL, and select the transcript language from the dropdown menu. For this case, choose Hindi, which matches the transcript language.

![step3](/Snapshots/step3.png)

- **📝 Step 04:** Click **Load and Process Transcript** to load the video transcript (this may take a few minutes depending on the video length). Then, click **Enter your preferred language** to choose your desired output language from the dropdown.

![step4](/Snapshots/step4.png)

After selecting your preferred answer language, click the **Prepare Mode** button to initialize the chatbot for processing.

![step4](/Snapshots/step4_1.png)

- **📝 Step 05:** Click **Enter your question about the video**, type your question, and then click **Get Answer**. The chatbot will display the answer, and the chat history updates with each new question.

![step5](/Snapshots/step5.png)

This process enables you to translate a YouTube video transcript from **Hindi to English** and interact with it through a chatbot interface. Users can load the transcript, translate it into their preferred language, and ask context-specific questions to receive accurate answers.

## 🎯 Test Case 02: Bengali ➡️ English

**Scenario:** Translating a *Nadir On The Go* video from Bengali to English with full transcript-based Q&A support.

| Snapshot 1 | Snapshot 2 |
|------------|------------|
| ![Nadir On The Go - Step 1](/Snapshots/Nadir%20-BE%20(1).png) | ![Nadir On The Go - Step 2](/Snapshots/Nadir%20-BE%20(2).png) |

| Snapshot 3 |
|------------|
| ![Nadir On The Go - Step 3](/Snapshots/Nadir%20-BE%20(3).png) |

[▶ **Watch the full video on YouTube**](https://youtu.be/snSfWYbenv8)


## 🎯 Test Case 03: English ➡️ Spanish

**Scenario:** Translating an Outdoor Boys video from English to Spanish with full transcript-based Q&A support.

| Snapshot 1 | Snapshot 2 |
|------------|------------|
| ![Outdoor Boys - Step 1](/Snapshots/outdoor%20boys.png) | ![Outdoor Boys - Step 2](/Snapshots/outdoor%20boys1.png) |

[▶ **Watch the full video on YouTube** ](https://youtu.be/MPo10rMzkNo)


## 🏆 Critical Challenges Faced & Solutions

This project faced several critical challenges while building. Here's how I have solved them:

---

### 1. 🔗 Handling Dynamic YouTube URLs
**Challenge:** Users may input invalid or malformed URLs, risking app crashes.  
**Solution:** ✅
- Used `extract_video_id()` with **regex validation**.  
- Wrapped processing in **try-except blocks**.  
- Displayed **clear error messages** with `st.error()`.

---

### 2. 🌐 Multilingual Support
**Challenge:** Users may want answers in a different language from the video’s original language. Translating long transcripts while keeping context is tricky.  
**Solution:** ✅
- Created `LANGUAGE_MAPPING` for dynamic source/target languages.  
- Split transcript into **2000-character chunks** for translation using `GoogleTranslator`.  
- Rejoined translated chunks and split into **1000-character chunks** for embedding with **100-character overlap** to preserve context.

---

### 3. 🧠 LLM Context & Embedding Limits
**Challenge:** Long transcripts exceed token limits for LLMs and embeddings.  
**Solution:** ✅
- **Multi-level chunking strategy**:
  1. First split: 2000-character chunks for translation.  
  2. Second split: 1000-character chunks for embeddings in **FAISS**.  
- Overlaps maintain context for accurate retrieval and LLM answers.

---

### 4. 📦 Session State Management in Streamlit
**Challenge:** Streamlit reruns scripts on each interaction, risking loss of **model, vector store, transcript, or chat history**.  
**Solution:** ✅
- Stored all key variables in `st.session_state`:
  - `model`, `vector_store`, `retriever`  
  - `split_transcript`, `chat_history`  
  - `source_language`, `target_language`, `video_url`  
- Ensures **smooth multi-step workflow** and persistent chat history.

---

### 5. ⏳ User Experience for Long-running Operations
**Challenge:** Transcript fetching, translation, and LLM processing can take time; users may think the app is frozen.  
**Solution:** ✅
- Used **`st.spinner()`** to provide **real-time feedback**:
  - `"Loading and processing transcript..."`
  - `"Generating answer..."`  
- Gives users a **responsive and friendly interface**.

---

> 💡 **Outcome:** The app can handle **any YouTube video**, support **multiple languages**, process **long transcripts efficiently**, preserve **session continuity**, and provide a **smooth user experience**.  

