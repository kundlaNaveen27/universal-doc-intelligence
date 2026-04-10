# vectorless_rag.py
# Two modes:
# 1. Full context — send everything (small docs)
# 2. Page index — build index, find relevant pages (medium docs)

import os
import json
import time
import hashlib
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# token limits
FULL_CONTEXT_LIMIT = 8000    # tokens — use full context
PAGE_INDEX_LIMIT   = 30000   # tokens — use page index
# above 30000 → use Traditional RAG


# ── METHOD 1: FULL CONTEXT ──────────────────────────
def answer_full_context(question, document_content, system_prompt):
    """
    Sends entire document to AI at once.
    Best for small documents under 8000 tokens.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Complete document:

{document_content}

Question: {question}
Answer based only on document. Cite page numbers."""
            }
        ],
        max_tokens=1000
    )

    answer = response.choices[0].message.content
    tokens = {
        "input": response.usage.prompt_tokens,
        "output": response.usage.completion_tokens,
        "total": response.usage.total_tokens
    }
    return answer, tokens


def get_cache_path(pdf_path):
    """
    Creates unique cache filename based on PDF content
    Different PDF = different cache file
    Same PDF = same cache file
    """
    # read PDF and create unique hash
    with open(pdf_path, 'rb') as f:
        pdf_hash = hashlib.md5(f.read()).hexdigest()[:8]
    
    return f"cache_{pdf_hash}_index.json"


def load_cache(pdf_path):
    """
    Loads existing index from cache file
    Returns None if no cache exists
    """
    cache_path = get_cache_path(pdf_path)
    
    if os.path.exists(cache_path):
        print(f"  📂 Loading index from cache: {cache_path}")
        with open(cache_path, 'r') as f:
            index = json.load(f)
        # convert keys back to integers
        index = {int(k): v for k, v in index.items()}
        print(f"  ✅ Loaded {len(index)} pages from cache — FREE!")
        return index
    
    return None  # no cache found


def save_cache(index, pdf_path):
    """
    Saves index to cache file for future use
    """
    cache_path = get_cache_path(pdf_path)
    with open(cache_path, 'w') as f:
        json.dump(index, f, indent=2)
    print(f"  💾 Index cached → {cache_path}")
    print(f"  ✅ Next time loads FREE from cache!")
# ── METHOD 2: PAGE INDEX ────────────────────────────
def build_page_index(pages_content, pdf_path=None):
    """
    Builds page index with caching.
    First time: calls API for each page, saves cache
    After that: loads from cache instantly (FREE)
    """
    # try loading from cache first
    if pdf_path:
        cached = load_cache(pdf_path)
        if cached:
            return cached

    print("  📑 Building page index (first time only)...")
    index = {}

    for page_num, content in pages_content.items():
        # skip nearly empty pages
        if len(content.strip()) < 100:
            print(f"    Page {page_num}: skipping (too short)")
            continue

        try:
            # wait to avoid rate limit
            time.sleep(2)

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Summarize this page in ONE sentence.
Be specific about the main topic and any 
key facts or numbers mentioned.

Page content:
{content[:800]}

One sentence summary:"""
                    }
                ],
                max_tokens=80
            )

            summary = response.choices[0].message.content.strip()
            index[page_num] = summary
            print(f"    Page {page_num}: {summary[:60]}...")

        except Exception as e:
            print(f"    Page {page_num}: error — {e}")
            index[page_num] = f"Page {page_num} content"
            time.sleep(5)  # longer wait on error
            continue

    # save to cache for future use
    if pdf_path:
        save_cache(index, pdf_path)

    print(f"  ✅ Index built for {len(index)} pages")
    return index


def find_relevant_pages(question, page_index, top_k=4):
    """
    Uses AI to find which pages are most relevant
    to the question using the page index.
    
    Returns list of relevant page numbers.
    """
    # format index for AI
    index_text = "\n".join([
        f"Page {num}: {summary}"
        for num, summary in page_index.items()
    ])

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"""Here is a document index:

{index_text}

Question: {question}

Which {top_k} pages are most likely to contain 
the answer? Return ONLY page numbers separated 
by commas. Example: 3, 7, 12, 15

Page numbers:"""
            }
        ],
        max_tokens=50
    )

    # parse page numbers from response
    raw = response.choices[0].message.content
    try:
        page_nums = []
        for part in raw.replace(" ", "").split(","):
            # extract digits only
            num = ''.join(filter(str.isdigit, part))
            if num:
                page_nums.append(int(num))
        return page_nums[:top_k]
    except Exception:
        # fallback — return first few pages
        return list(page_index.keys())[:top_k]


def answer_page_index(question, pages_content,
                      page_index, system_prompt):
    """
    Finds relevant pages using index
    Then answers from only those pages.
    Much cheaper than full context!
    """
    print(f"  🔍 Finding relevant pages for: '{question[:50]}...'")

    # find relevant pages
    relevant_page_nums = find_relevant_pages(
        question, page_index, top_k=4
    )
    print(f"  📄 Relevant pages found: {relevant_page_nums}")

    # build context from relevant pages only
    context = ""
    for page_num in relevant_page_nums:
        if page_num in pages_content:
            context += f"\n--- Page {page_num} ---\n"
            context += pages_content[page_num]
            context += "\n"

    if not context.strip():
        return "Could not find relevant pages.", {}

    estimated_tokens = len(context) // 4
    print(f"  📊 Sending ~{estimated_tokens:,} tokens (vs full doc)")

    time.sleep(3)

    # answer from relevant pages only
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Relevant pages from document:

{context}

Question: {question}

Answer based on the pages above.
Cite specific page numbers."""
            }
        ],
        max_tokens=1000
    )

    answer = response.choices[0].message.content
    tokens = {
        "input": response.usage.prompt_tokens,
        "output": response.usage.completion_tokens,
        "total": response.usage.total_tokens,
        "pages_used": relevant_page_nums
    }
    return answer, tokens


# ── MAIN FUNCTION — Auto selects method ─────────────
def answer_vectorless(question, document_content,
                      system_prompt, pages_content=None,
                      page_index=None):
    """
    Auto selects best method based on document size:
    → Small doc  (<8k tokens)  → Full context
    → Medium doc (<30k tokens) → Page index
    → Large doc  (>30k tokens) → Recommend Traditional RAG

    Args:
        question: user question
        document_content: full extracted text
        system_prompt: domain prompt
        pages_content: dict {page_num: page_text}
        page_index: pre-built index (optional)

    Returns:
        answer, tokens_used, method_used
    """
    token_estimate = len(document_content) // 4

    # Small document — use full context
    if token_estimate < FULL_CONTEXT_LIMIT:
        print(f"  🔵 Using Full Context method ({token_estimate:,} tokens)")
        answer, tokens = answer_full_context(
            question, document_content, system_prompt
        )
        return answer, tokens, "Full Context"

    # Medium document — use page index
    elif token_estimate < PAGE_INDEX_LIMIT:
        print(f"  🟡 Using Page Index method ({token_estimate:,} tokens)")

        # build index if not provided
        if page_index is None and pages_content:
            page_index = build_page_index(pages_content)
        elif page_index is None:
            # fallback to trimmed full context
            trimmed = document_content[:32000]
            answer, tokens = answer_full_context(
                question, trimmed, system_prompt
            )
            return answer, tokens, "Trimmed Full Context"

        answer, tokens = answer_page_index(
            question, pages_content,
            page_index, system_prompt
        )
        return answer, tokens, "Page Index"

    # Large document — recommend Traditional RAG
    else:
        return (
            "⚠️ Document too large for Vectorless RAG. "
            "Please use Traditional RAG mode for better results.",
            {},
            "Too Large"
        )


def is_suitable_for_vectorless(document_content):
    """
    Returns suitability message for UI
    """
    token_estimate = len(document_content) // 4

    if token_estimate < FULL_CONTEXT_LIMIT:
        return True, f"✅ Full Context mode (~{token_estimate:,} tokens)"
    elif token_estimate < PAGE_INDEX_LIMIT:
        return True, f"✅ Page Index mode (~{token_estimate:,} tokens)"
    else:
        return False, f"❌ Too large (~{token_estimate:,} tokens) — use Traditional RAG"


def extract_pages_dict(extraction_result):
    """
    Converts extractor result into pages dictionary
    {page_num: page_text}
    Needed for page index method
    """
    pages = {}
    for page_data in extraction_result["pages"]:
        page_num = page_data["page_number"]
        text = page_data.get("text", "")

        # add tables if any
        tables = page_data.get("tables", [])
        if tables:
            text += "\n" + "\n".join(tables)

        pages[page_num] = text

    return pages