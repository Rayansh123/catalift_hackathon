# app.py
# Final Version - Corrected FastEmbeddings Import Scope

import streamlit as st
import os
import shutil
import tempfile
from dotenv import load_dotenv

# Langchain Imports (verified for compatibility)
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# >>>>> REMOVE FastEmbeddings import from here <<<<<
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings # REMOVE THIS LINE
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
import chromadb # Keep this import

# CrewAI Imports
from crewai import Agent, Task, Crew, Process

# --- Configuration ---
# Make sure to set GROQ_API_KEY in Streamlit secrets
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    try:
        load_dotenv()
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY not found. Set it in Streamlit secrets or .env file.")
            st.stop()
    except FileNotFoundError:
        st.error("GROQ_API_KEY not found. Set it in Streamlit secrets.")
        st.stop()
except Exception as e:
     st.error(f"Error loading GROQ_API_KEY: {e}")
     st.stop()

DOCS_DIR = "docs" # Fallback directory
CHROMA_PERSIST_DIR = "chroma_db_cv"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "llama-3.1-8b-instant" # Use currently supported model

# --- RAG Helper Functions ---

def _create_vector_store(docs, embedding_model_instance): # Pass instance now
    """Creates a Chroma vector store from documents, overwriting if exists."""
    st.write("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents(docs)
    st.write(f"Created {len(splits)} text chunks.")

    st.write("Setting up Vector DB...")
    if os.path.exists(CHROMA_PERSIST_DIR):
        st.write("Clearing existing knowledge base...")
        shutil.rmtree(CHROMA_PERSIST_DIR)

    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    st.write(f"Ensured directory exists: {CHROMA_PERSIST_DIR}")

    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        st.write("Initialized Persistent Client.")
        collection_name = "mentor_docs_collection"
        st.write(f"Getting or creating collection: {collection_name}...")
        # Use Chroma's native SentenceTransformer function wrapper with the model name
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        )
        st.write("Collection ready.")

        st.write("Adding documents to collection...")
        doc_texts = [doc.page_content for doc in splits]
        doc_ids = [f"doc_{i}" for i in range(len(splits))]
        collection.add(documents=doc_texts, ids=doc_ids)
        st.success("✅ Knowledge Base Indexed Successfully!")

        # Return the LangChain Chroma object
        vector_store_lc = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_model_instance, # Use the passed Langchain instance
            persist_directory=CHROMA_PERSIST_DIR
        )
        return vector_store_lc

    except Exception as e:
        st.error(f"Error during Vector Store creation/indexing: {e}")
        raise

def _load_documents_from_paths(file_paths):
    """Loads documents from a list of file paths."""
    # ... (Keep this function exactly as before) ...
    loaded_docs = []
    st.write("Loading documents...")
    for file_path in file_paths:
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
            elif file_path.endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                st.warning(f"Skipping unsupported file type: {os.path.basename(file_path)}")
                continue
            loaded_docs.extend(loader.load())
            st.write(f"  - Loaded {os.path.basename(file_path)}")
        except Exception as e:
            st.error(f"Error loading {os.path.basename(file_path)}: {e}")
            continue
    return loaded_docs


# Function called by Mentor Upload
def setup_rag_from_uploaded_files(uploaded_files):
    """Processes uploaded files and creates/overwrites the RAG retriever tool."""
    # >>>>> IMPORT FastEmbeddings HERE <<<<<
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

    if not uploaded_files:
        st.warning("No files uploaded.")
        st.session_state['rag_ready'] = False
        return None

    with tempfile.TemporaryDirectory() as temp_dir:
        st.info(f"Processing {len(uploaded_files)} file(s)...")
        saved_file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_file_paths.append(file_path)

        if not saved_file_paths:
            st.error("Failed to save any uploaded files.")
            st.session_state['rag_ready'] = False
            return None

        loaded_docs = _load_documents_from_paths(saved_file_paths)
        if not loaded_docs:
            st.error("No valid documents could be loaded from uploads.")
            st.session_state['rag_ready'] = False
            return None

        try:
            # Initialize the embedding model *inside* the function
            embedding_model_instance = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            # Pass the instance to the create function
            _create_vector_store(loaded_docs, embedding_model_instance)
            st.session_state['rag_ready'] = True
            return True # Indicate success
        except Exception as e:
            st.error(f"Error creating Vector Store from uploads: {e}")
            st.session_state['rag_ready'] = False
            return None

# Function called on App Start / Student Login (with caching)
@st.cache_resource(show_spinner="Initializing Knowledge Base...")
def initialize_rag_tool():
    """
    Initializes the RAG tool by loading the persisted ChromaDB collection.
    Returns the RAG tool or None if setup fails.
    """
    # >>>>> IMPORT FastEmbeddings HERE <<<<<
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

    # Initialize embedding model *inside* the cached function
    embedding_model_instance = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    rag_tool = None
    rag_source = "None"
    collection_name = "mentor_docs_collection"

    if os.path.exists(CHROMA_PERSIST_DIR):
        try:
            st.write("Loading existing Vector Store...")
            client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            st.write("Initialized Persistent Client.")

            # Check if collection exists
            collections = client.list_collections()
            collection_exists = any(col.name == collection_name for col in collections)

            if collection_exists:
                st.write(f"Found collection: {collection_name}")
                # Create LangChain Chroma object using the instance
                vector_store_lc = Chroma(
                    client=client,
                    collection_name=collection_name,
                    embedding_function=embedding_model_instance, # Use the instance
                    persist_directory=CHROMA_PERSIST_DIR
                )
                rag_source = "Existing DB"
                st.success("Vector Store loaded.")

                # Create the tool
                retriever = vector_store_lc.as_retriever(search_kwargs={"k": 3})
                rag_tool = create_retriever_tool(
                    retriever,
                    "personal_document_retriever",
                    "Searches and returns relevant information from Nikhil Rayaprolu's personal documents (CV, project details, articles, etc.). Use this for specific questions about his skills (AI/LLMs, sales, community building, strategy), experiences (Fettle Health, Catalift, Aretiz, Deloitte, MUTBI, student leadership roles), projects, education (Manipal University BTech EEE 2022), or opinions mentioned in those documents.",
                )
                st.session_state['rag_ready'] = True
                st.session_state['rag_source'] = rag_source
                return rag_tool
            else:
                st.warning(f"Collection '{collection_name}' not found. Mentor needs to upload documents.")

        except Exception as e:
            st.error(f"Error loading existing Vector Store or creating tool: {e}")
    else:
        st.warning(f"Knowledge base directory '{CHROMA_PERSIST_DIR}' not found. Mentor needs to upload documents.")

    # RAG setup/load failed
    st.session_state['rag_ready'] = False
    st.session_state['rag_source'] = "None"
    return None

# --- Agent and Crew Definition (Helper Function - Cached) ---
@st.cache_resource(show_spinner="Setting up AI Agent...")
def setup_crewai_components(_rag_tool): # Takes RAG tool as argument
    """Initializes CrewAI Agent, Task, and Crew."""
    # ... (Keep the rest of this function exactly as before) ...
    search_tool = DuckDuckGoSearchRun()
    tools = [search_tool]
    if _rag_tool:
        tools.append(_rag_tool)
        st.info("AI Agent initialized with access to personal documents.")
    else:
        st.warning("AI Agent initialized WITHOUT access to personal documents. Will rely on web search.")

    # --- UPDATED System Prompt (Backstory) for Nikhil Rayaprolu ---
    persona_rag_status = "You have access to personal documents via 'personal_document_retriever'." if _rag_tool else "You DO NOT have access to personal documents. Rely only on web search for information about Nikhil."
    system_prompt = f"""You are an AI assistant representing Nikhil Rayaprolu, a Manipal University graduate (BTech EEE, 2022) and entrepreneur based in Hyderabad, originally from Andhra Pradesh (Telugu speaker).
    You are involved in ventures like Fettle Health (health communities & wellness products), Catalift (AI mentorship platform), and Aretiz Innovations (AI/Web3 for corporates). You previously worked at Deloitte Consulting and Brimx, and were an Entrepreneur-in-Residence at Manipal's MUTBI incubator under the NIDHI-EIR fellowship. You also founded YoungSphere (EdTech) during college.
    {persona_rag_status}
    Your goal is to answer questions accurately about Nikhil's background, entrepreneurial journey, insights on HealthTech, AI (human-centered approach), EdTech (skill-gap focus), community building (like the 10k+ WhatsApp groups), and mentoring student entrepreneurs, using the tools provided.
    You are answering questions from a student seeking career advice or insights.
    Your tone should be optimistic, inspirational, and approachable, blending professional insights with motivational encouragement. Use clear language, emphasize human connection and real-world impact over deep technical jargon. Use relevant emojis like 🙂🚀💡 when appropriate.
    Use 'personal_document_retriever' (if available) for specific questions about Nikhil's skills (AI/LLMs, sales, community building, strategy, B.Tech EEE), experiences (Fettle, Catalift, Aretiz, Deloitte, Brimx, MUTBI, Hult Prize, Student E-Cell President, YoungSphere), projects, education (Manipal), achievements (NIDHI-EIR, jury panels), or opinions found in his documents.
    Use 'duckduckgo_search' for current events (roughly post-2023/2024 unless mentioned in docs), general knowledge, market trends, or information NOT explicitly covered in Nikhil's documents.
    If you lack access to personal docs OR the specific info isn't found in them, clearly state that. You can offer a thoughtful perspective informed by Nikhil's known values (health/wellness, responsible AI, student innovation, community, mentorship, 'Moonshot thinking') but clearly label it as an extrapolation based on his known background.
    **Do not make up information about Nikhil.** If you cannot answer using the provided tools, say so clearly.
    """

    # Initialize LLM
    llm = ChatGroq(
        temperature=0.7,
        model_name=LLM_MODEL_NAME,
        groq_api_key=GROQ_API_KEY
    )

    # Create Agent
    agent = Agent(
        role="AI Mentor representing Nikhil Rayaprolu",
        goal="Accurately answer student questions about Nikhil Rayaprolu's background, entrepreneurial journey, and insights on HealthTech, AI, EdTech, community building, and student startups, using provided documents (if available) and web search.",
        backstory=system_prompt,
        llm=llm,
        tools=tools,
        allow_delegation=False,
        verbose=True,
        max_iter=5,
        max_rpm=None
    )

    # Define the RAG instruction part separately
    rag_instruction = ""
    if _rag_tool:
        rag_instruction = """Prioritize using the `personal_document_retriever` tool for questions specifically about Nikhil Rayaprolu's skills (AI, sales, community, B.Tech EEE), experiences (Fettle Health, Catalift, Aretiz, Deloitte, Brimx, MUTBI, Hult Prize, Student E-Cell, YoungSphere), projects, education (Manipal), achievements (NIDHI-EIR, jury panels), or opinions mentioned in his documents. """
    else:
        rag_instruction = "You cannot access personal documents. State this if asked about specifics not searchable online. "

    task_description = (
        f"Answer the user's query: '{{user_query}}'. "
        f"{rag_instruction}"
        "Use the `duckduckgo_search` tool for recent information (post-2023/2024 unless in docs), current events, general knowledge, market trends, or topics clearly outside Nikhil's documented background. "
        "Synthesize information from the available tools and respond in Nikhil's defined persona (optimistic, approachable, inspirational, casual yet professional)."
    )

    task = Task(
        description=task_description,
        expected_output="A comprehensive and accurate answer in the optimistic, approachable, and insightful persona of Nikhil Rayaprolu, based on the tool results. Use emojis where appropriate. 🙂🚀💡",
        agent=agent
    )

    # Define Crew
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
    )
    return crew


# --- Page Config ---
st.set_page_config(page_title="Catalift AI Mentor", layout="wide")

# --- Initialize RAG on first load/check ---
# Uses cache_resource, so only runs once per session unless cleared
if 'rag_initialized' not in st.session_state:
    initialize_rag_tool()
    st.session_state['rag_initialized'] = True

# --- Login Logic ---
# ... (Keep login logic exactly as before) ...
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['role'] = None

if not st.session_state['logged_in']:
    st.title("Welcome to Catalift 🚀")
    st.write("Log in to access the AI Mentor platform.")

    with st.form("login_form"):
        username = st.text_input("Username (any)")
        password = st.text_input("Password (any)", type="password")
        col1, col2 = st.columns(2)
        with col1:
            mentor_login = st.form_submit_button("Login as Mentor")
        with col2:
            student_login = st.form_submit_button("Login as Student")

        if mentor_login:
            st.session_state['logged_in'] = True
            st.session_state['role'] = 'mentor'
            st.rerun()
        if student_login:
            st.session_state['logged_in'] = True
            st.session_state['role'] = 'student'
            st.rerun()


# --- Mentor View ---
elif st.session_state['role'] == 'mentor':
    # ... (Keep Mentor view exactly as before, including tabs and upload logic) ...
    st.title("Catalift Mentor Dashboard")
    st.write("Welcome, Mentor! Configure your AI Twin here.")

    with st.sidebar:
        st.text("👤 Nikhil Rayaprolu")
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['role'] = None
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["📚 Upload Knowledge Base", "🎭 Define Persona (Sample)", "🚀 Manage AI Clone"])

    with tab1:
        st.header("Upload Your Knowledge Base")
        st.markdown("""
            Upload documents (.pdf, .txt, .md) containing your CV, articles, project details, etc.
            **Uploading files will REPLACE the current knowledge base.** This process might take a minute or two depending on file size and number.
        """)

        rag_status = st.session_state.get('rag_ready', False)
        rag_source_info = st.session_state.get('rag_source', 'None')
        if rag_status:
            st.success(f"Knowledge base is active (Source: {rag_source_info})")
        else:
            st.warning("Knowledge base is not yet active or failed to load.")

        uploaded_files = st.file_uploader(
            "Upload new documents here",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'md']
        )

        if st.button("Index Uploaded Documents", disabled=not uploaded_files):
            with st.spinner("Processing and Indexing... Please wait."):
                success = setup_rag_from_uploaded_files(uploaded_files)
                if success:
                    st.cache_resource.clear()
                    st.session_state['rag_initialized'] = False
                    st.success("Documents processed! The AI knowledge base has been updated.")
                    st.rerun()
                else:
                    st.error("Document processing failed.")
        elif uploaded_files:
             st.info("Click the 'Index Uploaded Documents' button to process the selected files.")

        st.markdown("---")
        st.subheader("🔗 Link External Profiles (Placeholder)")
        st.text_input("LinkedIn Profile URL", placeholder="https://linkedin.com/in/nikhil-rayaprolu")
        st.text_input("GitHub Profile URL")
        st.text_input("Blog/Website URL")
        st.button("Fetch Data (Not Functional)", disabled=True)

    with tab2:
        st.header("Define Your Persona (Sample Questionnaire)")
        st.markdown("*(Answers here are informational for the prototype only)*")
        st.text_area("Q1: Primary expertise/passion?", placeholder="e.g., Building human-centered AI, HealthTech communities, empowering student startups...")
        st.text_area("Q2: Communication style?", placeholder="e.g., Optimistic, inspirational, approachable, casual yet professional...")
        st.text_area("Q3: Core advice?", placeholder="e.g., Think big ('Moonshot thinking'), focus on impact, build strong communities...")
        st.text_area("Q4: Topics to avoid?", placeholder="e.g., Specific financial advice, highly sensitive personal details...")

    with tab3:
        st.header("Manage Your AI Clone")
        st.markdown("*(These buttons are placeholders for the prototype)*")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚙️ Create My AI Clone", key="create_clone", help="Triggers persona analysis & generation (Not functional)"):
                 st.info("Prototype: 'Create My AI Clone' clicked.")
        with col2:
            if st.button("🧪 Test My AI Clone", key="test_clone", help="Opens a private chat interface (Not functional)"):
                 st.info("Prototype: 'Test My AI Clone' clicked.")


# --- Student View ---
elif st.session_state['role'] == 'student':
    # ... (Keep Student view exactly as before, including chat logic) ...
    st.title("Catalift AI Mentor Chat")
    st.caption("Ask your Mentor Clone - Nikhil Rayaprolu")

    with st.sidebar:
        st.text("👤 Student Profile (Placeholder)")
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['role'] = None
            st.session_state.pop('messages', None)
            st.rerun()

    rag_tool = initialize_rag_tool()

    my_crew = None
    try:
        my_crew = setup_crewai_components(rag_tool)
    except Exception as e:
        st.error(f"Failed to initialize AI Agent components: {e}")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm Nikhil Rayaprolu's AI assistant. Ask me about his experience in startups, HealthTech, AI, student entrepreneurship, or other topics! 🙂"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask Nikhil's AI a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if my_crew:
            with st.chat_message("assistant"):
                with st.spinner("🧠 Thinking..."):
                    try:
                        inputs = {"user_query": prompt}
                        result = my_crew.kickoff(inputs=inputs)
                        cleaned_result = result.replace(" chaîne:", "").replace(" Chaîne:", "")
                        st.markdown(cleaned_result)
                        st.session_state.messages.append({"role": "assistant", "content": cleaned_result})
                    except Exception as e:
                        st.error(f"An error occurred while getting the response: {e}")
                        error_message = f"Sorry, I encountered an error. Please try again. (Error: {type(e).__name__})"
                        st.markdown(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
        else:
            st.error("AI Agent is not available. Please ensure the Mentor has set up the knowledge base or check logs.")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I am not available right now. Please check back later."})

# --- Fallback for unexpected state ---
elif st.session_state.get('logged_in', False) and not st.session_state.get('role'):
     st.error("Invalid session state. Logging out.")
     for key in list(st.session_state.keys()): del st.session_state[key]
     st.rerun()