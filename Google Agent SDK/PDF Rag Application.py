import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


def rag_chat_ui():
    st.set_page_config(layout="wide")

    load_dotenv()
    os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

    # --- Session State Initialization ---
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {"Default": []}
        st.session_state.current_session = "Default"
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    # --- CSS for Scrollable Columns ---
    st.markdown("""
        <style>
            .scrollable-col {
                max-height: 90vh;
                overflow-y: auto;
                padding-right: 1rem;
            }
            .block-container {
                padding-top: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 2])

    # === Column A: Sessions ===
    with col1:
        st.markdown('<div class="scrollable-col">', unsafe_allow_html=True)

        st.markdown("### 🗂️ Sessions")
        for session_name in st.session_state.chat_sessions:
            if st.button(session_name, key=session_name):
                st.session_state.current_session = session_name

        new_session = st.text_input("Create session", key="new_session_input")
        if st.button("➕ Add"):
            if new_session and new_session not in st.session_state.chat_sessions:
                st.session_state.chat_sessions[new_session] = []
                st.session_state.current_session = new_session

        st.markdown('</div>', unsafe_allow_html=True)

    # === Column B: Document Upload + Sources ===
    with col2:
        st.markdown('<div class="scrollable-col">', unsafe_allow_html=True)

        st.header("📄 Document Tools")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")

        if st.button("📊 Analyze Document"):
            if uploaded_file:
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    loader = PyPDFLoader(uploaded_file.name)
                    documents = loader.load()

                    vector_store = Chroma.from_documents(
                        documents=documents,
                        embedding=OpenAIEmbeddings(),
                        persist_directory="./chroma_db",
                        collection_name="Research-Paper-Prediction"
                    )

                    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                    rag_chain = RetrievalQA.from_chain_type(
                        llm=ChatOpenAI(model="gpt-4"),
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                        verbose=True
                    )

                    st.session_state.vector_store = vector_store
                    st.session_state.retriever = retriever
                    st.session_state.rag_chain = rag_chain

                    st.success("✅ Document analyzed and RAG chain ready!")

                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                finally:
                    if os.path.exists(uploaded_file.name):
                        os.remove(uploaded_file.name)

        # === Show Sources Only per Question ===
        messages = st.session_state.chat_sessions[st.session_state.current_session]
        if messages:
            st.subheader("📚 Sources Used")
            question_count = 0
            for idx, entry in enumerate(messages):
                if entry[0] == "user":
                    question_count += 1
                    user_question = entry[1]

                    # Find matching assistant response with sources
                    if idx + 1 < len(messages) and messages[idx + 1][0] == "assistant":
                        assistant_entry = messages[idx + 1]
                        sources = assistant_entry[2] if len(assistant_entry) > 2 else []

                        with st.expander(f"Q{question_count}: {user_question[:60]}..."):
                            if sources:
                                for src in sources:
                                    st.write(f"- {src}")
                            else:
                                st.write("- No sources found.")

        st.markdown('</div>', unsafe_allow_html=True)

    # === Column C: Chat ===
    with col3:
        st.markdown('<div class="scrollable-col">', unsafe_allow_html=True)

        st.header(f"💬 Chat - {st.session_state.current_session}")
        messages = st.session_state.chat_sessions[st.session_state.current_session]

        for chat in reversed(messages):
            if chat[0] == "assistant":
                role, msg = chat[0], chat[1]
            else:
                role, msg = chat
            icon = "🧠" if role == "assistant" else "🙋"
            st.markdown(f"**{icon} {role.capitalize()}**: {msg}", unsafe_allow_html=True)

        # Chat Input with form to handle Enter key
        with st.form(key='chat_form'):
            user_input = st.text_input("Your message:", value="", key="chat_input")
            submit_button = st.form_submit_button("📨 Send")

            if submit_button and user_input.strip():
                messages.append(("user", user_input))

                if st.session_state.rag_chain:
                    try:
                        result = st.session_state.rag_chain.invoke({"query": user_input})
                        answer = result["result"]
                        sources = list(set(
                            [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
                        ))
                        messages.append(("assistant", answer, sources))
                    except Exception as e:
                        messages.append(("assistant", f"Error: {str(e)}", []))
                else:
                    messages.append(("assistant", "Please analyze a document first.", []))

                # Clear the input
                st.session_state.chat_input = ""
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# Run the app
if __name__ == "__main__":
    rag_chat_ui()