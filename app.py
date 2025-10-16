import streamlit as st
import os
import glob
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# Page config
st.set_page_config(
    page_title="Drone Manual QA System | RAG Application",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    .header-subtitle {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        text-align: center;
        opacity: 0.95;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid;
    }
    .user-message {
        background-color: #ffffff;
        border-left-color: #667eea;
    }
    .bot-message {
        background-color: #f8f9fa;
        border-left-color: #764ba2;
    }
    .message-label {
        font-weight: 600;
        font-size: 0.9rem;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    .status-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .status-ready {
        background-color: #d4edda;
        border-left-color: #28a745;
        color: #155724;
    }
    .status-pending {
        background-color: #fff3cd;
        border-left-color: #ffc107;
        color: #856404;
    }
    .status-error {
        background-color: #f8d7da;
        border-left-color: #dc3545;
        color: #721c24;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    .sidebar .element-container {
        margin-bottom: 1rem;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #667eea30;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 1px solid #dee2e6;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

class DroneQASystem:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.docs = []
        
    def load_documents(self):
        """Load PDF and TXT documents from data folder"""
        self.docs = []
        
        pdf_files = glob.glob("data/*drone_manual.pdf")
        for file_path in pdf_files:
            loader = PyPDFLoader(file_path)
            self.docs.extend(loader.load())
        
        txt_files = glob.glob("data/*.txt")
        for file_path in txt_files:
            loader = TextLoader(file_path)
            self.docs.extend(loader.load())
        
        return len(pdf_files), len(txt_files), len(self.docs)
    
    def split_documents(self):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150
        )
        split_docs = text_splitter.split_documents(self.docs)
        return split_docs
    
    def create_vectorstore(self, split_docs):
        """Create FAISS vectorstore"""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = FAISS.from_documents(split_docs, embeddings)
        self.vectorstore.save_local("vectorstore")
        return self.vectorstore
    
    def load_existing_vectorstore(self):
        """Load existing vectorstore"""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = FAISS.load_local(
            "vectorstore", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    
    def setup_qa_chain(self, model_name="phi"):
        """Setup QA chain with Ollama"""
        llm = Ollama(model=model_name)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )
    
    def ask_question(self, question):
        """Ask a question"""
        result = self.qa_chain.invoke({"query": question})
        return {
            'answer': result['result'],
            'sources': result.get('source_documents', [])
        }

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = DroneQASystem()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False
if 'metrics' not in st.session_state:
    st.session_state.metrics = {'docs': 0, 'chunks': 0, 'queries': 0}

# Header
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">Drone Manual QA System</h1>
        <p class="header-subtitle">Retrieval-Augmented Generation (RAG) for Intelligent Document Query</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## Configuration Panel")
    
    # Create data folder if not exists
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Document Management
    st.markdown("### Document Management")
    pdf_files = glob.glob("data/*.pdf")
    txt_files = glob.glob("data/*.txt")
    
    if pdf_files or txt_files:
        st.markdown("**Loaded Documents:**")
        for f in pdf_files:
            st.text(f"PDF: {Path(f).name}")
        for f in txt_files:
            st.text(f"TXT: {Path(f).name}")
    else:
        st.markdown('<div class="status-card status-pending">No documents found. Please upload files.</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File upload
    st.markdown("### Upload Documents")
    uploaded_files = st.file_uploader(
        "Select PDF or TXT files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload drone manuals or technical documentation"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Successfully uploaded {len(uploaded_files)} file(s)")
    
    st.markdown("---")
    
    # Model configuration
    st.markdown("### Model Configuration")
    model_choice = st.selectbox(
        "LLM Model",
        ["phi", "tinyllama", "llama2", "gemma:2b"],
        help="Ensure the model is installed via Ollama"
    )
    
    use_existing = st.checkbox(
        "Use existing vectorstore",
        help="Load pre-computed embeddings to skip reprocessing"
    )
    
    # Build pipeline
    if st.button("Initialize RAG Pipeline"):
        try:
            if use_existing and os.path.exists("vectorstore"):
                with st.spinner("Loading vectorstore..."):
                    st.session_state.qa_system.load_existing_vectorstore()
                    st.session_state.qa_system.setup_qa_chain(model_choice)
                    st.session_state.system_ready = True
                    st.success("Vectorstore loaded successfully")
            else:
                with st.spinner("Loading documents..."):
                    pdf_count, txt_count, doc_count = st.session_state.qa_system.load_documents()
                    
                    if doc_count == 0:
                        st.error("No documents found in data directory")
                        st.stop()
                    
                    st.session_state.metrics['docs'] = pdf_count + txt_count
                    st.success(f"Loaded {doc_count} document pages")
                
                with st.spinner("Processing text chunks..."):
                    split_docs = st.session_state.qa_system.split_documents()
                    st.session_state.metrics['chunks'] = len(split_docs)
                    st.success(f"Created {len(split_docs)} text chunks")
                
                with st.spinner("Building vector embeddings..."):
                    st.session_state.qa_system.create_vectorstore(split_docs)
                    st.success("Vector embeddings created")
                
                with st.spinner(f"Initializing {model_choice} model..."):
                    st.session_state.qa_system.setup_qa_chain(model_choice)
                    st.success("QA chain initialized")
                
                st.session_state.system_ready = True
            
            st.balloons()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            if "Connection" in str(e):
                st.markdown('<div class="status-card status-error">Ollama server not running. Execute: <code>ollama serve</code></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System status
    st.markdown("### System Status")
    if st.session_state.system_ready:
        st.markdown('<div class="status-card status-ready"><strong>Status:</strong> Ready for queries</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-card status-pending"><strong>Status:</strong> Awaiting initialization</div>', unsafe_allow_html=True)
    
    # Metrics
    if st.session_state.metrics['docs'] > 0:
        st.markdown("### Pipeline Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", st.session_state.metrics['docs'])
        with col2:
            st.metric("Chunks", st.session_state.metrics['chunks'])
        st.metric("Queries Processed", st.session_state.metrics['queries'])
    
    st.markdown("---")
    
    # Technical details
    with st.expander("Technical Implementation"):
        st.markdown("""
        **Architecture:**
        - Embeddings: HuggingFace (all-MiniLM-L6-v2)
        - Vector Store: FAISS
        - LLM: Ollama (Local deployment)
        - Framework: LangChain
        
        **Processing:**
        - Chunk Size: 1000 tokens
        - Overlap: 150 tokens
        - Retrieval: Top-3 relevant chunks
        """)
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.metrics['queries'] = 0
        st.rerun()

# Main interface
if st.session_state.system_ready:
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Conversation History")
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <div class="message-label">USER QUERY</div>
                        <div>{msg['content']}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div class="message-label">SYSTEM RESPONSE</div>
                        <div>{msg['content']}</div>
                    </div>
                """, unsafe_allow_html=True)
                if 'sources' in msg and msg['sources']:
                    with st.expander(f"View {len(msg['sources'])} source references"):
                        for i, source in enumerate(msg['sources'], 1):
                            st.markdown(f"**Reference {i}:**")
                            st.text(source.page_content[:300] + "...")
                            if hasattr(source, 'metadata'):
                                st.caption(f"Page: {source.metadata.get('page', 'N/A')}")
                            st.markdown("---")
    
    # Query input
    st.markdown("---")
    st.markdown("### New Query")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        question = st.text_input(
            "Enter your question",
            key="question_input",
            placeholder="Example: What is the maximum flight altitude?",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_button = st.button("Submit", use_container_width=True)
    
    # Suggested queries
    st.markdown("**Sample Queries:**")
    cols = st.columns(3)
    suggestions = [
        "How do I calibrate the compass?",
        "What is the maximum flight altitude?",
        "How do I perform return to home?"
    ]
    
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion, key=f"suggest_{i}", use_container_width=True):
            question = suggestion
            ask_button = True
    
    # Process query
    if ask_button and question:
        st.session_state.chat_history.append({
            'role': 'user',
            'content': question
        })
        
        with st.spinner("Processing query..."):
            try:
                result = st.session_state.qa_system.ask_question(question)
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'sources': result['sources']
                })
                
                st.session_state.metrics['queries'] += 1
                st.rerun()
                
            except Exception as e:
                st.error(f"Query processing failed: {str(e)}")

else:
    # Welcome interface
    st.markdown("""
    <div class="info-box">
        <h3>Welcome to the Drone Manual QA System</h3>
        <p>An advanced Retrieval-Augmented Generation (RAG) application for intelligent document querying.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Setup Instructions
        
        **Prerequisites:**
        1. Install Ollama
        2. Pull required model
        ```bash
        ollama pull phi
        ```
        3. Start Ollama server
        ```bash
        ollama serve
        ```
        
        **Application Setup:**
        1. Upload drone manuals (PDF/TXT)
        2. Select LLM model
        3. Click "Initialize RAG Pipeline"
        4. Begin querying
        """)
    
    with col2:
        st.markdown("""
        ### Key Features
        
        - **Multi-format Support**: PDF and TXT documents
        - **Vector Search**: FAISS-powered semantic retrieval
        - **Local LLM**: Ollama integration for privacy
        - **Source Attribution**: Track answer provenance
        - **Persistent Storage**: Reusable vector embeddings
        - **Interactive UI**: Real-time query processing
        
        ### Use Cases
        - Technical documentation queries
        - Maintenance procedure lookup
        - Specification verification
        - Troubleshooting assistance
        """)
    
    st.markdown("""
    ### Example Queries
    - "How do I calibrate the compass?"
    - "What is the maximum flight altitude?"
    - "How do I perform return to home?"
    - "What are the battery specifications?"
    - "How to update the firmware?"
    """)

# Footer
st.markdown("""
    <div class="footer">
        <strong>Drone Manual QA System</strong> | RAG Application<br>
        Built with LangChain • Ollama • FAISS • Streamlit<br>
        <em>Intelligent Document Retrieval for Agentic AI Systems</em>
    </div>
""", unsafe_allow_html=True)