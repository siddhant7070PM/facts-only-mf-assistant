import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# Retrieve Groq API Key safely from Streamlit Secrets or Environment Variables
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

st.set_page_config(page_title="Facts-Only MF Assistant", page_icon="📈")
st.title("📈 Facts-Only MF Assistant (Groww)")
st.caption("Facts-only. No investment advice. Last updated from sources: Oct 2026")

st.markdown("""
**Try asking:**
* *What is the expense ratio of SBI Bluechip Fund?*
* *Is there a lock-in period for the ELSS fund?*
* *What is the exit load for SBI Flexicap?*
* *How can I download my capital gains statement?*
""")

@st.cache_resource
def load_rag_model():
    loader = TextLoader("knowledge_base.txt")
    docs = loader.load()
    
    # 1. Embeddings remain on the lightweight MiniLM model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # 2. ChatGroq points to the active 2026 model
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b", 
        temperature=0.1,
        groq_api_key=groq_api_key
    )
    
    prompt_template = """
    You are a factual Mutual Fund assistant. 
    Rules:
    1. Answer using ONLY the provided context. If the answer is not in the context, say "I don't have that information in my verified sources."
    2. Do NOT provide investment advice. If the user asks for opinions (e.g., "should I buy/sell?"), reply: "I can only provide factual data. Please consult a financial advisor for investment decisions. See AMFI's educational portal: https://www.amfiindia.com/investor-corner"
    3. Keep answers strictly under 3 sentences.
    4. You MUST include the exact "Source:" URL from the context at the end of your answer.
    
    Context: {context}
    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )

if not groq_api_key:
    st.error("Groq API Key is missing. Please add GROQ_API_KEY to your Streamlit Cloud Secrets.")
else:
    chain = load_rag_model()

    user_query = st.text_input("Ask a factual question about SBI Mutual Fund schemes:")
    if user_query:
        with st.spinner("Searching verified sources..."):
            try:
                response = chain.run(user_query)
                st.write(response)
            except Exception as e:
                st.error(f"API Connection Error: {str(e)}")