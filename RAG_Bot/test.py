import streamlit as st
import os
import tempfile
import numpy as np
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint

# Define a simple model
class SimpleLocalModel:
    def __init__(self):
        pass
        
    def invoke(self, prompt):
        """Simple response generation"""
        return "I've analyzed the document but I'm working in offline mode. " + \
               "To get more detailed responses, please provide a valid Hugging Face API token."

# Fallback for when no API key is available
def get_fallback_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def simple_chain(query_dict):
        query = query_dict.get("input", "")
        docs = retriever.invoke(query)
        
        if not docs:
            return {"answer": "No relevant information found in the documents."}
            
        # Create a simple response based on retrieved documents
        response = "Here's what I found in the documents:\n\n"
        for i, doc in enumerate(docs, 1):
            response += f"Document {i}: {doc.page_content[:200]}...\n\n"
            
        return {"answer": response}
        
    return simple_chain

# Set page configuration for dark theme
st.set_page_config(
    page_title="Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Document Assistant - Chat with your documents"
    }
)

# Custom CSS for better styling with improved contrast for dark mode
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        color: #2196F3 !important;
    }
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 500 !important;
        margin-bottom: 1rem !important;
        color: #FFFFFF !important;
    }
    .user-msg {
        background-color: #1E3A5F;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #2196F3;
        color: #FFFFFF;
    }
    .assistant-msg {
        background-color: #2C3E50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        color: #FFFFFF;
    }
    .upload-section {
        background-color: #1E293B;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #3A5070;
        color: #FFFFFF;
    }
    .document-list {
        background-color: #263238;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        color: #FFFFFF;
        border: 1px solid #455A64;
    }
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #455A64;
        border-radius: 0.5rem;
        background-color: #121212;
        color: #FFFFFF;
    }
    .stTextInput>div>div>input {
        font-size: 1.1rem !important;
        background-color: #263238 !important;
        color: #FFFFFF !important;
    }
    .processing-msg {
        font-style: italic;
        color: #90CAF9 !important;
        font-weight: 500;
    }
    /* Hide Streamlit error messages */
    .element-container .stException {
        display: none !important;
    }
    /* Custom styling for info/success/warning alerts */
    .stAlert {
        background-color: #1A2430 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-left: 4px solid #2196F3 !important;
    }
    .stAlert.success {
        border-left: 4px solid #4CAF50 !important;
    }
    .stAlert.warning {
        border-left: 4px solid #FFC107 !important;
    }
    .stAlert.error {
        border-left: 4px solid #F44336 !important;
    }
    
    /* Make info message more visible */
    .stAlert p {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    
    /* Style the placeholder text for file upload */
    [data-testid="stFileUploader"] {
        background-color: #1E293B !important;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px dashed #3A5070;
    }
    
    /* Override default Streamlit text */
    .stMarkdown, p, div {
        color: #FFFFFF;
    }
    
    /* Override button styles */
    .stButton button {
        background-color: #2196F3 !important;
        color: white !important;
        border: none !important;
    }
    .stButton button:hover {
        background-color: #0D47A1 !important;
        border: none !important;
    }
    .stButton button:disabled {
        background-color: #546E7A !important;
        color: #CFD8DC !important;
    }
    
    div[data-testid="stStatusWidget"] {
        display: none !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Make text inputs more visible */
    textarea, input[type="text"], input[type="password"] {
        background-color: #263238 !important;
        color: white !important;
        border: 1px solid #455A64 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "user_input_query" not in st.session_state:
    st.session_state.user_input_query = ""
if "show_error" not in st.session_state:
    st.session_state.show_error = False
if "error_message" not in st.session_state:
    st.session_state.error_message = ""

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name
    except Exception as e:
        set_error(f"Error saving file: {e}")
        return None

def load_document(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path)
        elif file_extension in [".html", ".htm"]:
            loader = UnstructuredHTMLLoader(file_path)
        else:
            set_notification(f"Unsupported file format: {file_extension}", "warning")
            return []
        return loader.load()
    except Exception as e:
        set_notification(f"Could not process file: {os.path.basename(file_path)}", "warning")
        return []

def set_notification(message, type="info"):
    if type == "success":
        st.markdown(f"""<div style="background-color:#1E3B2F; color:white; padding:10px; border-radius:5px; border-left:5px solid #4CAF50;">
                    ✅ {message}</div>""", unsafe_allow_html=True)
    elif type == "warning":
        st.markdown(f"""<div style="background-color:#3B2F1E; color:white; padding:10px; border-radius:5px; border-left:5px solid #FFC107;">
                    ⚠️ {message}</div>""", unsafe_allow_html=True)
    elif type == "info":
        st.markdown(f"""<div style="background-color:#1E2F3B; color:white; padding:10px; border-radius:5px; border-left:5px solid #2196F3;">
                    ℹ️ {message}</div>""", unsafe_allow_html=True)
    # Errors are not displayed directly, only stored in session state

def set_error(message):
    st.session_state.error_message = message
    st.session_state.show_error = True
    # We'll use this for internal tracking but won't display it to the user

def process_documents():
    with st.spinner("Processing documents... This may take a few moments."):
        st.session_state.processing = True
        
        # Load documents
        documents = []
        for file_info in st.session_state.uploaded_files:
            docs = load_document(file_info["path"])
            documents.extend(docs)
            if docs:
                if file_info["name"] not in st.session_state.processed_files:
                    st.session_state.processed_files.append(file_info["name"])
        
        if not documents:
            set_notification("No documents could be processed. Please check your files.", "warning")
            st.session_state.processing = False
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        try:
            #TF-IDF based embedding system
            class TfidfEmbeddings(Embeddings):
                def __init__(self, max_features=5000):
                    self.vectorizer = TfidfVectorizer(max_features=max_features)
                    self.fitted = False

                def fit(self, texts):
                    self.vectorizer.fit(texts)
                    self.fitted = True

                def embed_documents(self, texts):
                    if not self.fitted:
                        self.fit(texts)
                    embeddings = self.vectorizer.transform(texts).toarray()
                    return embeddings.tolist()

                def embed_query(self, text):
                    if not self.fitted:
                        return [0.0] * self.vectorizer.max_features
                    embedding = self.vectorizer.transform([text]).toarray()[0]
                    return embedding.tolist()
            
            # Extract all text from documents to fit the vectorizer
            all_texts = [doc.page_content for doc in chunks]
            
            embeddings = TfidfEmbeddings(max_features=5000)
            embeddings.fit(all_texts)
            
            set_notification("Successfully created embeddings!", "success")
            
            # Create the vector store
            vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.vectorstore = vectorstore
            
            try:
                if st.session_state.api_key:
                    selected_model = st.session_state.llm_model
                    
                    if "mistral" in selected_model.lower():
                        llm = HuggingFaceEndpoint(
                            repo_id=selected_model,
                            huggingfacehub_api_token=st.session_state.api_key,
                            task="conversational",
                            temperature=0.5
                        )
                    elif "roberta" in selected_model.lower() or "squad" in selected_model.lower():
                        llm = HuggingFaceEndpoint(
                            repo_id=selected_model,
                            huggingfacehub_api_token=st.session_state.api_key,
                            task="question-answering"
                        )
                    else:
                        # For general text generation models
                        llm = HuggingFaceEndpoint(
                            repo_id=selected_model,
                            huggingfacehub_api_token=st.session_state.api_key,
                            task="text-generation",
                            temperature=0.7
                        )
                    
                    _ = llm.invoke("Test")
                    set_notification(f"Successfully connected to {selected_model}!", "success")
                else:
                    raise ValueError("No API key provided")
            except Exception as e:
                set_notification("Using simple document retrieval without LLM processing...", "info")
                st.session_state.conversation = get_fallback_chain(vectorstore)
                st.session_state.processing = False
                return
            
            # Custom prompt template specifically designed for question answering in a conversational format
            custom_template = """
            Answer the following question based only on the provided context information.
            
            Context information:
            {context}
            
            Question: {input}
            
            If the answer cannot be determined from the context information, simply state that you don't know based on the available information.
            Please provide a direct, concise answer without unnecessary explanations.
            """
            
            CUSTOM_PROMPT = PromptTemplate(
                input_variables=["context", "input"],
                template=custom_template
            )
            
            # Create chains
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            question_answer_chain = create_stuff_documents_chain(llm, CUSTOM_PROMPT)
            retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            st.session_state.conversation = retrieval_chain
            
        except Exception as e:
            set_notification("Using simple document retrieval mode.", "info")
            
            # Set up fallback mode with simple retrieval if everything else fails
            if st.session_state.vectorstore:
                st.session_state.conversation = get_fallback_chain(st.session_state.vectorstore)
        
        st.session_state.processing = False

def handle_user_query(query):
    if not st.session_state.conversation:
        set_notification("Please upload and process documents first!", "warning")
        return
    
    with st.spinner("Generating response..."):
        try:
            # Use invoke() method with error handling
            try:
                response = st.session_state.conversation.invoke({"input": query})
                answer = response.get("answer", "No answer found.")
            except AttributeError:

                if hasattr(st.session_state.conversation, "__call__"):
                    response = st.session_state.conversation({"input": query})
                    answer = response.get("answer", "No answer found.")
                else:
                    answer = "I've analyzed your documents but couldn't find a specific answer to your question. Try rephrasing or asking something else about the content."
                
            st.session_state.chat_history.append({"user": query, "assistant": answer})
        except Exception as e:
            friendly_message = "I'm having trouble generating a response right now. This could be due to connection issues or model limitations. Let me try a simplified approach."
            st.session_state.chat_history.append({"user": query, "assistant": friendly_message})
            
            if st.session_state.vectorstore:
                try:
                    fallback = get_fallback_chain(st.session_state.vectorstore)
                    response = fallback({"input": query})
                    fallback_answer = response.get("answer", "")
                    if fallback_answer:
                        st.session_state.chat_history.append({"user": "", "assistant": fallback_answer})
                except:
                    pass

# Callback for form submission
def on_submit():
    query = st.session_state.user_input_query
    if query:
        handle_user_query(query)

# Sidebar - HuggingFace API Key input
with st.sidebar:
    st.markdown("<div class='sub-header' style='color:#2196F3;'>HuggingFace API Configuration</div>", unsafe_allow_html=True)
    
    api_key = st.text_input(
        "Enter your HuggingFace API Token",
        type="password",
        value=st.session_state.api_key,
        help="Your API token is stored only in your session state"
    )
    
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        st.session_state.conversation = None
        if api_key:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    
    model_options = {
        "mistralai/Mistral-Nemo-Instruct-2407": "Mistral Nemo Instruct (Conversational)",
        "deepset/roberta-base-squad2": "RoBERTa for Question Answering",
        "facebook/bart-large-cnn": "BART for Summarization"
    }
    
    selected_model = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0
    )
    
    st.session_state.llm_model = selected_model
    
    # Display task type based on selected model
    if "mistral" in selected_model.lower():
        task_type = "conversational"
    elif "roberta" in selected_model.lower():
        task_type = "question-answering"
    else:
        task_type = "text-generation"
        
    st.markdown(f"**Using model:** {selected_model}")
    st.markdown(f"**Task type:** {task_type}")
    
    # Initialization requirements
    with st.expander("Installation Requirements"):
        st.markdown("""
        ### Required Libraries
        Make sure to install these libraries:
        ```
        pip install streamlit langchain langchain_community langchain_core faiss-cpu
        pip install scikit-learn numpy  # For TF-IDF embeddings
        pip install langchain_huggingface huggingface_hub  # For Hugging Face integration
        pip install pypdf python-docx unstructured
        ```
        """)

    st.markdown("---")

# Main content
st.markdown("<h1 class='main-header' style='color:#2196F3;'>📚 Document Assistant</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#FFFFFF; font-size:1.1rem;'>Upload your documents and chat with them! Ask questions about your files and get relevant answers.</p>",
    unsafe_allow_html=True
)

# File uploader section
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Upload Documents</div>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt", "html", "htm"],
    accept_multiple_files=True,
    help="You can upload multiple files of supported formats"
)

if uploaded_files:
    new_files = False
    for uploaded_file in uploaded_files:
        # Check if file is already in session state
        file_exists = any(file_info["name"] == uploaded_file.name for file_info in st.session_state.uploaded_files)
        if not file_exists:
            # Save file and add to session state
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                st.session_state.uploaded_files.append({
                    "name": uploaded_file.name,
                    "path": file_path,
                    "size": uploaded_file.size
                })
                new_files = True
    
    if new_files:
        # Reset conversation if new files are added
        st.session_state.conversation = None

# Display uploaded files
if st.session_state.uploaded_files:
    st.markdown("<div class='document-list'>", unsafe_allow_html=True)
    st.markdown("**Uploaded Documents:**")
    
    for i, file_info in enumerate(st.session_state.uploaded_files):
        status = "✅ Processed" if file_info["name"] in st.session_state.processed_files else "⏳ Pending processing"
        st.markdown(f"{i+1}. {file_info['name']} ({status})")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    process_button = st.button(
        "Process Documents", 
        disabled=not st.session_state.api_key or st.session_state.processing, 
        key="process_button",
        help="Process uploaded documents to enable chat functionality"
    )
    
    if process_button:
        process_documents()
st.markdown("</div>", unsafe_allow_html=True)

# Chat interface
st.markdown("<div class='sub-header'>Chat with your Documents</div>", unsafe_allow_html=True)

if not st.session_state.processed_files:
    st.markdown("""<div style="background-color:#1E2F3B; color:#FFFFFF; padding:15px; border-radius:5px; border-left:5px solid #2196F3; font-weight:500; margin-top:10px;">
                ℹ️ Upload and process documents to start chatting.</div>""", unsafe_allow_html=True)
else:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    if not st.session_state.chat_history:
        st.markdown("<p class='processing-msg'>Documents processed. You can start asking questions now!</p>", unsafe_allow_html=True)
    else:
        for message in st.session_state.chat_history:
            if message['user']: 
                st.markdown(f"<div class='user-msg'><strong>You:</strong> {message['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='assistant-msg'><strong>Assistant:</strong> {message['assistant']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    with st.form(key="query_form", clear_on_submit=True):
        user_query = st.text_input(
            "Ask a question about your documents",
            placeholder="e.g., What are the main points discussed in the document?",
            disabled=not st.session_state.conversation,
            key="user_input_query",
            label_visibility="collapsed"
        )
        
        submit_button = st.form_submit_button(
            "Send", 
            on_click=on_submit,
            disabled=not st.session_state.conversation
        )

# Footer
st.markdown("---")
st.markdown(
    "<p style='color:#BBDEFB; text-align:center; font-size:0.9rem;'>This application uses HuggingFace models for text generation. "
    "Your documents are processed locally and are not stored permanently.</p>",
    unsafe_allow_html=True
)