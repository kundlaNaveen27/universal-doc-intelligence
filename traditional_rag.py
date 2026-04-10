# traditional_rag.py
# Uses Pinecone vector database
# Best for large documents and high query volume
# Cheapest per query (~375 tokens)

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# ── EMBEDDING CLASS ──────────────────────────────────
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


# ── PINECONE SETUP ───────────────────────────────────
def setup_pinecone_index(index_name="universal-doc-intelligence"):
    """
    Creates Pinecone index if not exists
    Returns the index
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing:
        print(f"  Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("  ✅ Index created!")
    else:
        print(f"  ✅ Using existing index: {index_name}")

    return pc.Index(index_name)


def index_document(pdf_path, filename=None,
                   index_name="universal-doc-intelligence"):
    
    # use provided filename or extract from path
    display_name = filename or os.path.basename(pdf_path)
    print(f"\n  📥 Indexing {display_name} into Pinecone...")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = splitter.split_documents(pages)

    # use display_name not tmp path
    for chunk in chunks:
        chunk.metadata["source"] = display_name

    print(f"  📄 Created {len(chunks)} chunks")

    embeddings = SentenceTransformerEmbeddings()
    print("  🔢 Creating embeddings and storing...")

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )

    print(f"  ✅ {len(chunks)} chunks stored in Pinecone!")
    return vectorstore

# ── CONNECT TO EXISTING INDEX ────────────────────────
def connect_to_index(index_name="universal-doc-intelligence"):
    """
    Connects to existing Pinecone index
    Use this after document is already indexed
    """
    embeddings = SentenceTransformerEmbeddings()
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    return vectorstore


# ── ANSWER QUESTION ──────────────────────────────────
def answer_traditional(question, vectorstore,
                       system_prompt, top_k=3):
    """
    Searches Pinecone for relevant chunks
    Sends only relevant chunks to AI
    Most token-efficient method

    Args:
        question: user question
        vectorstore: connected Pinecone vectorstore
        system_prompt: domain-specific prompt
        top_k: number of chunks to retrieve

    Returns:
        answer, sources, tokens_used
    """
    print(f"  🔍 Searching Pinecone for relevant chunks...")

    # search Pinecone
    results = vectorstore.similarity_search(
        question, k=top_k
    )

    if not results:
        return "No relevant information found.", [], {}

    # build context from chunks
    context = ""
    sources = []

    for i, doc in enumerate(results):
        context += f"\nSection {i+1}:\n{doc.page_content}\n"
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        if page != "?":
            page = int(float(page))
        sources.append(f"{source} (page {page})")

    print(f"  ✅ Found {len(results)} relevant chunks")
    estimated_tokens = len(context) // 4
    print(f"  📊 Sending ~{estimated_tokens} tokens to AI")

    # connect to Groq
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

    # answer from relevant chunks only
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Context from document:
{context}

Question: {question}

Answer based only on the context above.
Cite which section your answer comes from.""")
    ]

    response = llm.invoke(messages)
    answer = response.content

    tokens = {
        "input": response.usage_metadata.get(
            "input_tokens", 0),
        "output": response.usage_metadata.get(
            "output_tokens", 0),
        "total": response.usage_metadata.get(
            "total_tokens", 0)
    }

    return answer, sources, tokens


# ── CLEAR INDEX ──────────────────────────────────────
def clear_index(index_name="universal-doc-intelligence"):
    """
    Deletes and recreates Pinecone index
    Use when you want to start fresh
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    existing = [idx.name for idx in pc.list_indexes()]

    if index_name in existing:
        print(f"  🗑️ Deleting index: {index_name}")
        pc.delete_index(index_name)
        print("  ✅ Deleted!")

    return setup_pinecone_index(index_name)