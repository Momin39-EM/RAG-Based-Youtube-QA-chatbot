import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from deep_translator import GoogleTranslator

# ===== Language Mapping =====
LANGUAGE_MAPPING = {
    "English": "en",
    "Bengali": "bn",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
    "Chinese": "zh",
    "Japanese": "ja",
}

YOUTUBE_STATIC_PROMPT = """
You are a helpful assistant that answers questions based on a YouTube video transcript.

Source Language: {source_language}
Target Language: {target_language}

If the transcript is in a different language than the target language, translate your final answer into the target language.

Provide clear, concise, and correct answers.
Transcript context:
{context}
"""


# ===== Extract Video ID =====
def extract_video_id(url):
    """Extracts the video ID from a YouTube URL."""
    import re
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if not match:
        raise ValueError("Invalid YouTube URL.")
    return match.group(1)

def process_transcript(video_id, lang_code, preferred_language):
    """Fetches, translates, and splits transcript."""
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=[lang_code])
        transcript = " ".join(chunk.text for chunk in transcript_list)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        split_transcript = text_splitter.split_text(transcript)

        translator = GoogleTranslator(source='auto', target=preferred_language)
        translated_chunks = [translator.translate(chunk) for chunk in split_transcript]
        translated_text = " ".join(translated_chunks)

        text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_transcript_final = text_splitter2.split_text(translated_text)

        return split_transcript_final
    except Exception as e:
        raise RuntimeError(f"Error in processing transcript: {e}")

def create_vector_store(split_transcript_final):
    """Creates a FAISS vector store from the split transcript."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        docs = [Document(page_content=chunk) for chunk in split_transcript_final]
        vector_store = FAISS.from_documents(docs, embedding=embeddings)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {e}")

def create_retriever_chain(vector_store, model, source_language=None, target_language=None):
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        # Prepare prompt with dynamic source/target language info
        prompt_template = YOUTUBE_STATIC_PROMPT
        
        # Format prompt template with the languages to create the actual prompt
        prompt_text = prompt_template.format(
            source_language=source_language or "English",
            target_language=target_language or "English",
            context="{context}"  # Keep {context} as a placeholder for documents
        )
        
        # Create ChatPromptTemplate from the formatted prompt text
        prompt = ChatPromptTemplate.from_template(prompt_text)

        document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
        retriever_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

        return retriever_chain
    except Exception as e:
        raise RuntimeError(f"Failed to create retriever chain: {e}")


def generate_response(retriever, question, source_language='English', target_language='English'):
    try:
        response = retriever.invoke({
            "input": question,
            "context": "",   # Context will be automatically provided by retriever internally, you can pass empty here
            "source_language": source_language,
            "target_language": target_language
        })
        return response.get('answer', 'No answer generated.')
    except Exception as e:
        raise RuntimeError(f"Failed to generate response: {e}")

def instantiate_model(api_key):
    """Loads the Groq LLM."""
    return ChatGroq(
        temperature=0,
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key
    )

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    model = instantiate_model(api_key)

    url = input("🔗 Enter the YouTube video URL: ").strip()
    video_id = extract_video_id(url)

    print("\n🌐 Available languages:", ", ".join(LANGUAGE_MAPPING.keys()))
    source_language = input("📹 Enter the video's language: ").strip()
    target_language = input("📝 Enter your preferred answer language: ").strip()

    if source_language not in LANGUAGE_MAPPING:
        raise ValueError(f"Invalid source language: {source_language}")

    if target_language not in LANGUAGE_MAPPING:
        raise ValueError(f"Invalid target language: {target_language}")

    lang_code = LANGUAGE_MAPPING[source_language]
    preferred_language = LANGUAGE_MAPPING[target_language]

    print(f"\n✅ Source Language code: {lang_code}")
    print(f"✅ Target Language code: {preferred_language}")

    split_transcript = process_transcript(video_id, lang_code, preferred_language)
    vector_store = create_vector_store(split_transcript)
    retriever = create_retriever_chain(vector_store, model, lang_code, preferred_language)

    print("\nYou can now ask questions about the video transcript. Type 'exit' to quit.")
    while True:
        question = input("\n❓ Enter your question: ").strip()
        if question.lower() == "exit":
            print("Exiting. Goodbye!")
            break
        try:
            answer = generate_response(retriever, question, lang_code, preferred_language)
            print(f"\n💬 Answer ({target_language}): {answer}")
        except Exception as e:
            print(f"Error generating response: {e}")