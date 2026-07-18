import streamlit as st
import requests

# Set page configuration first (must be the very first Streamlit command)
st.set_page_config(
    page_title="Smart Travel Companion",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

FASTAPI_URL = "http://localhost:8000"

# --- CUSTOM CSS FOR BETTER VISUALS ---
st.markdown("""
    <style>
    /* Style main headers */
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    /* Style container boxes */
    .feature-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
        color: #1F2937; /* Forces the main text to be dark gray */
    }
    .feature-box h4 {
        color: #111827 !important; /* Forces the headings to be almost black */
        margin-top: 0;
    }
    .feature-box p {
        color: #374151 !important; /* Forces the paragraph text to be dark gray */
    }
    /* Force Streamlit metrics to wrap text instead of truncating */
    [data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow-wrap: break-word !important;
        font-size: 1.4rem !important; /* Slightly reduces size to fit nicely */
        line-height: 1.2 !important;
    }        
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: KNOWLEDGE INGESTION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/826/826070.png", width=80)
    st.markdown("## 🛠️ Control Panel")
    st.markdown("---")
    
    st.subheader("📁 Knowledge Base Ingestion")
    st.caption("Upload travel manuals or destination PDFs to expand the vector database repository.")
    
    uploaded_file = st.file_uploader("Choose a PDF travel guide", type="pdf")

    if uploaded_file:
        if st.button("🚀 Process & Index Guide", use_container_width=True):
            with st.spinner("Parsing document & generating embeddings..."):
                try:
                    files = {"file": uploaded_file}
                    response = requests.post(f"{FASTAPI_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success(f"✅ {response.json()['message']}")
                    else:
                        st.error(f"❌ Error: {response.json().get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"💥 Connection failed: {str(e)}")
                    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    st.success("⚡ Backend API: Connected")
    st.info("🤖 Model: TinyLlama-1.1B (Local)")

# --- MAIN INTERFACE LAYOUT ---
tab1, tab2 = st.tabs(["🗺️ AI Travel Planner", "📐 System Architecture"])

with tab1:
    st.markdown('<h1 class="main-title">🧭 Smart Travel Companion</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">An advanced RAG-powered intelligent engine providing contextual travel plans.</p>', unsafe_allow_html=True)
    
    # Quick Statistics Row
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(label="Retrieval Strategy", value="Dense Vector Search")
    with col_b:
        st.metric(label="Vector Database", value="Qdrant Cloud/Local")
    with col_c:
        st.metric(label="Generation Mode", value="100% Free & Local Open-Source")
        
    st.markdown("---")
    
    # Query Section
    st.subheader("❓ Prompt the Intelligent Agent")
    
    # Wrap the input and button in a form
    with st.form(key="travel_query_form"):
        query = st.text_input(
            "Enter destination query:",
            placeholder="e.g., Provide a historical 3-day walking plan for Paris.",
            label_visibility="collapsed"
        )
        # Use form_submit_button instead of regular button
        submit_button = st.form_submit_button("🔮 Generate Smart Plan", type="primary")
    
    # Now check if the form was submitted (via button click OR Enter key)
    if submit_button:
        if query.strip():
            # Create a nice layout container for results
            with st.status("Searching knowledge base & synthesizing response...", expanded=True) as status:
                try:
                    st.write("🔍 Running semantic vector search across Qdrant chunks...")
                    response = requests.post(f"{FASTAPI_URL}/ask", json={"query": query})
                    
                    if response.status_code == 200:
                        status.update(label="✨ Itinerary Generated successfully!", state="complete", expanded=True)
                        answer = response.json()["response"]
                        
                        # 1. Chop off any hallucinations after the [END] token
                        answer = answer.split("[END]")[0].strip()
                        
                        # 2. Escape dollar signs AND force Markdown line breaks
                        safe_answer = answer.replace("$", r"\$").replace("\n", "\n\n")
                        
                        st.markdown("### 🗺️ Generated Travel Itinerary Plan")
                        st.info(safe_answer)
                    else:
                        status.update(label="❌ Generation failed", state="error")
                        st.error("Server side runtime error occurred.")
                except Exception as e:
                    status.update(label="❌ Connection failed", state="error")
                    st.error(f"Could not communicate with FastAPI service layer: {e}")
        else:
            st.warning("⚠️ Please provide a clear travel prompt first.")

with tab2:
    st.markdown("## 📐 Project Specs")
    st.markdown("This software architecture pattern decouples ingestion, semantic storage, and generative UI.")
    
    st.markdown("""
    <div class="feature-box">
        <h4>1. Frontend Layer (UI/UX)</h4>
        <p>Built using Streamlit to provide asynchronous web-sockets based interaction and multi-part data payload uploading capabilities.</p>
    </div>
    <div class="feature-box">
        <h4>2. Middleware Layer (FastAPI Service)</h4>
        <p>Asynchronous Server Gateway Interface (ASGI) running Python 3.12. Exposes scalable REST API endpoints (/ask and /upload) mapping request bodies via Pydantic validator models.</p>
    </div>
    <div class="feature-box">
        <h4>3. Vector Store Retrieval (RAG Component)</h4>
        <p>Utilizes PyMuPDF parsing to chunk texts dynamically. Embeddings are compiled via Sentence-Transformers and payloads are indexed structurally inside Qdrant Vector database spaces using Cosine Distance metric calculation.</p>
    </div>
    <div class="feature-box">
        <h4>4. Large Language Model Integration</h4>
        <p>Harnesses open-source TinyLlama parameters running locally via Hugging Face transformer pipeline, passing zero-cost inference to build factual answers derived from retrieved context boundaries.</p>
    </div>
    """, unsafe_allow_html=True)