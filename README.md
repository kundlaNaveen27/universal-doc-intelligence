# 🧠 Universal Document Intelligence System

AI-powered document analysis platform that works across 
ANY industry — healthcare, legal, financial, insurance, 
and government. Handles text, tables, images, charts, 
and math formulas.

## 🚀 Live Demo
👉 [Try it here](https://universal-doc-intelligence-jjwy8tntwibwrpbzfpng9e.streamlit.app/)

## The Problem
Every industry deals with complex documents:
- Hospitals: patient records with lab tables and medical images
- Banks: financial reports with charts and data tables  
- Law firms: contracts with structured clauses
- Insurance: policies with coverage tables

Existing solutions either ignore images/tables OR cost 
$500k+ to implement. This system handles everything for free.

## What Makes This Different
Most RAG projects handle text only. This system handles:
- ✅ Text extraction (pdfplumber)
- ✅ Table structure preservation
- ✅ Image extraction and AI description (Vision AI)
- ✅ Vector graphics detection and analysis
- ✅ False positive filtering (NO_FIGURE system)
- ✅ Two RAG modes for different document sizes

## Two RAG Modes

### Vectorless RAG
- Full Context: sends entire doc to AI (small docs)
- Page Index: builds smart index, sends only relevant 
  pages (medium docs) — 7x cheaper than full context
- Cache system: index built once, reused forever FREE

### Traditional RAG  
- Pinecone vector database
- Semantic chunk search
- ~375 tokens per query (10x cheaper than vectorless)
- Best for large documents and high query volume

## Auto Mode Selection
```
< 8,000 tokens  → Vectorless Full Context
< 30,000 tokens → Vectorless Page Index (with cache)
> 30,000 tokens → Traditional RAG (Pinecone)
```

## 6 Domain-Specific Modes
| Domain | Optimized For |
|--------|--------------|
| 🏥 Medical | Clinical notes, lab results, diagnoses |
| ⚖️ Legal | Contracts, obligations, deadlines |
| 💰 Financial | Revenue, ratios, compliance risks |
| 🏦 Insurance | Coverage, exclusions, claims |
| 🏛️ Government | Regulations, compliance, policy |
| 📦 General | Any document type |

## Tech Stack
- **pdfplumber** — text and table extraction
- **PyMuPDF** — image extraction
- **PIL** — image format conversion
- **LLaMA Vision** — image understanding
- **Pinecone** — cloud vector database
- **LangChain** — RAG orchestration
- **SentenceTransformers** — embeddings
- **Groq + LLaMA 3.3 70B** — AI responses
- **Streamlit** — web interface

## Architecture
```
PDF Upload
    ↓
extractor.py — handles ALL content types
    ├── Text (pdfplumber)
    ├── Tables (pdfplumber - structure preserved)
    ├── Bitmap images (PyMuPDF + Vision AI)
    └── Vector graphics (PyMuPDF + render + Vision AI)
    ↓
domains.py — sector-specific AI prompting
    ↓
RAG Engine (auto-selected):
    ├── vectorless_rag.py (full context or page index)
    └── traditional_rag.py (Pinecone vector search)
    ↓
app.py — Streamlit UI with chat interface
```

## Setup
```bash
pip install -r requirements.txt
```

Add to `.env`:
```
GROQ_API_KEY=your_key
PINECONE_API_KEY=your_key
```

Run:
```bash
streamlit run app.py
```

## ⚠️ Current Limitations
- Small decorative icons may fail Vision AI processing
- Math formulas extracted as text (LaTeX not preserved)
- Scanned PDFs need OCR (pytesseract)
- Handwritten notes not supported

## 🔮 Future Improvements
- LlamaParse for professional-grade extraction
- Azure Document Intelligence for HIPAA compliance
- OCR support for scanned documents
- LangGraph agent for multi-step reasoning
- Conversation memory across sessions

## Real World Application
This architecture mirrors enterprise document intelligence
systems used by major banks, hospitals, and law firms —
built here as an open-source alternative.

## What I Learned
- Multimodal PDF extraction (text + tables + images)
- False positive filtering in Vision AI pipelines
- Vectorless RAG with page indexing and caching
- Traditional RAG with Pinecone
- Domain-specific prompt engineering
- Auto RAG mode selection based on document size
- Token optimization strategies
