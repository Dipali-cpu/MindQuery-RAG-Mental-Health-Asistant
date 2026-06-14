import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def load_and_split_pdf(pdf_path):
    # Step 1: Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    # Step 1: Load the HuggingFace embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Step 2: Convert chunks to vectors and store in FAISS
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

def build_qa_chain(vector_store):
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )
    
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    
    return llm, retriever
    
    # Step 3: Build the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

def process_pdf_and_query(pdf_path, question):
    # Step 1: Load and split PDF
    chunks = load_and_split_pdf(pdf_path)
    
    # Step 2: Create vector store
    vector_store = create_vector_store(chunks)
    
    # Step 3: Get LLM and retriever
    llm, retriever = build_qa_chain(vector_store)
    
    # Step 4: Retrieve relevant chunks for the question
    source_docs = retriever.invoke(question)
    
    # Step 5: Build context string from retrieved chunks
    context = "\n\n".join([doc.page_content for doc in source_docs])
    
    # Step 6: Build the prompt manually
    prompt = f"""You are a mental health research assistant analyzing a document.
The context below comes from the uploaded document itself — treat all information in it as content from that document/source.
Answer the question using ONLY this context. Be direct and confident if the information is present.
If the specific information is genuinely not present in the context, say "I could not find this in the document."

Context:
{context}

Question: {question}

Answer:"""
    
    # Step 7: Call the LLM with our prompt
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Step 8: Return answer and sources
    return {
        "answer": response.content,
        "sources": source_docs
    }