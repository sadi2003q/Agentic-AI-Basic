import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime
import time

load_dotenv()

# Initialize session state variables
if 'all_sessions' not in st.session_state:
    st.session_state.all_sessions = {}
if 'current_session' not in st.session_state:
    st.session_state.current_session = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = False
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )


def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode


def apply_dark_mode():
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
            .stApp {
                background-color: #1E1E1E;
                color: white;
            }
            .stTextInput>div>div>input {
                color: white;
                background-color: #333333;
            }
            .css-1d391kg {
                background-color: #1E1E1E;
            }
            .stChatMessage {
                background-color: #2D2D2D;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .stApp {
                background-color: white;
                color: black;
            }
            .stChatMessage {
                background-color: #F0F2F6;
            }
        </style>
        """, unsafe_allow_html=True)


def get_pdf_reader(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunk):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_texts(text_chunk, embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.session_state.processed_pdfs = True


def get_conversational_chain():
    prompt_template = """
    You are an AI assistant for answering questions about research papers.
    You are given the following extracted parts of a research paper and a question.
    Provide a detailed answer that combines information from the paper with our conversation history.

    Conversation History:
    {chat_history}

    Paper Context:
    {context}

    Question: {question}

    Answer in detail, and if the answer isn't in the paper, say "This information isn't available in the paper." 
    but you can still use our conversation history to help answer.
    """

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"]
    )

    # Create conversational chain with memory
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=FAISS.load_local(
            "faiss_index",
            OpenAIEmbeddings(model="text-embedding-3-large"),
            allow_dangerous_deserialization=True
        ).as_retriever(),
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return chain


def user_input(user_question):
    if not st.session_state.processed_pdfs:
        st.warning("Please process PDFs first!")
        return "PDFs not processed yet."

    try:
        # Check if this question was already asked in current session
        for q, a in st.session_state.current_session:
            if q == user_question:
                return a  # Return cached answer

        chain = get_conversational_chain()
        response = chain({"question": user_question})

        answer = response["answer"]
        sources = [doc.metadata for doc in response["source_documents"]]

        # Store in current session
        st.session_state.current_session.append((user_question, answer))

        # Update all sessions with current timestamp
        timestamp = datetime.now().strftime("Session %Y-%m-%d %H:%M:%S")
        st.session_state.all_sessions[timestamp] = st.session_state.current_session.copy()

        # Add source information if available
        if sources:
            answer += "\n\nSources:\n"
            for i, source in enumerate(sources, 1):
                if 'page' in source:
                    answer += f"{i}. Page {source['page']}\n"
                else:
                    answer += f"{i}. Section {source.get('source', 'unknown')}\n"

        return answer

    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        return f"Error: {str(e)}"


def main():
    st.set_page_config(page_title="PDF Chat Assistant", layout="wide")
    apply_dark_mode()

    # Sidebar
    with st.sidebar:
        st.title("☰ Menu")

        # Dark mode toggle with callback
        st.toggle("🌙 Dark Mode",
                  value=st.session_state.dark_mode,
                  key="dark_mode_toggle",
                  on_change=toggle_dark_mode)

        menu_option = st.radio("Navigation", ["Chat", "History"])

        if menu_option == "History":
            st.subheader("📜 Chat Sessions")
            session_names = list(st.session_state.all_sessions.keys())

            # Display session names without triggering reload
            selected_session = st.selectbox("Select a session",
                                            [""] + session_names,
                                            index=0)

            if selected_session:
                st.session_state.current_session = st.session_state.all_sessions[selected_session].copy()
                # Clear memory and reinitialize for the selected session
                st.session_state.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key='answer'
                )
                # Rebuild memory from selected session
                for q, a in st.session_state.current_session:
                    st.session_state.memory.save_context(
                        {"input": q},
                        {"output": a}
                    )
                st.rerun()

        # Add button to clear current conversation
        if st.button("🧹 Clear Current Conversation"):
            st.session_state.current_session = []
            st.session_state.memory.clear()
            st.rerun()

    # Main content
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📄 Upload PDF")
        pdf_docs = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)

        if pdf_docs and st.button("📚 Process PDFs"):
            with st.spinner("Processing and indexing..."):
                raw_text = get_pdf_reader(pdf_docs)
                text_chunks = get_text_chunk(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Done! Ask your questions now.")

    with col2:
        st.markdown("### 💬 Chat")

        # Display current chat
        for q, a in st.session_state.current_session:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)

        # Question input
        if prompt := st.chat_input("Type your question..."):
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = user_input(prompt)
                    st.write(answer)


if __name__ == "__main__":
    main()