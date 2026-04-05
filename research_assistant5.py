"""
LangGraph Multi-Agent Research Assistant
Single-file Streamlit application with full orchestration pipeline.
"""

import os
import json
import time
import uuid
import hashlib
import traceback
from datetime import datetime
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

# ─── Page config must be first ────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Intelligence System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dependency checks ────────────────────────────────────────────────────────
MISSING = []
try:
    import requests
except ImportError:
    MISSING.append("requests")

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from typing_extensions import TypedDict, Annotated
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    MISSING.append("langgraph")

try:
    import wikipedia
    HAS_WIKI = True
except ImportError:
    HAS_WIKI = False

try:
    import arxiv
    HAS_ARXIV = True
except ImportError:
    HAS_ARXIV = False

try:
    import faiss
    import PyPDF2
    import numpy as np
    HAS_RAG = True
except ImportError:
    HAS_RAG = False

if MISSING:
    st.error(f"Missing packages: `{', '.join(MISSING)}`. Run: `pip install {' '.join(MISSING)} langgraph wikipedia arxiv`")
    st.stop()

# ─── Typed state ──────────────────────────────────────────────────────────────
from typing_extensions import TypedDict, Annotated
from typing import List, Dict


class ResearchState(TypedDict):
    query: str
    sub_questions: List[Dict]
    selected_path: str
    investigation_paths: List[Dict]
    retrieval_results: List[Dict]
    rag_results: List[Dict]          # PDF RAG hits
    analysis: Dict
    insights: Dict
    red_team: Dict
    gap_fill_results: List[Dict]     # iterative gap-fill retrieval
    final_report: Dict
    audit_trail: List[Dict]
    status: str
    error: Optional[str]
    iteration: int
    token_count: int                 # running token estimate


# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --ink: #0f0e0d;
    --paper: #f7f4ef;
    --amber: #d4871a;
    --rust: #b84c2e;
    --sage: #3d6b52;
    --slate: #3a4a5c;
    --mist: #e8e4dc;
    --gold: #c9a227;
    /* card/surface colours that flip in dark mode */
    --card-bg: #ffffff;
    --card-border: var(--mist);
    --snippet-color: #555555;
    --even-row-bg: #faf8f5;
    --validator-bg: #fafaf7;
    --validator-border: #e0dcd5;
    --warn-bg: #fff7ed;
    --conf-bar-bg: #eeeeee;
}

/* Dark-mode overrides using Streamlit's data attribute */
[data-theme="dark"] {
    --ink: #f0ece3;
    --paper: #1a1917;
    --slate: #9baabf;
    --mist: #2e2c29;
    --card-bg: #242220;
    --card-border: #3a3835;
    --snippet-color: #aaaaaa;
    --even-row-bg: #1f1d1b;
    --validator-bg: #1e1c1a;
    --validator-border: #3a3835;
    --warn-bg: #2a1f10;
    --conf-bar-bg: #333333;
}

/* Also target Streamlit's auto dark class */
@media (prefers-color-scheme: dark) {
    :root {
        --ink: #f0ece3;
        --paper: #1a1917;
        --slate: #9baabf;
        --mist: #2e2c29;
        --card-bg: #242220;
        --card-border: #3a3835;
        --snippet-color: #aaaaaa;
        --even-row-bg: #1f1d1b;
        --validator-bg: #1e1c1a;
        --validator-border: #3a3835;
        --warn-bg: #2a1f10;
        --conf-bar-bg: #333333;
    }
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--paper);
    color: var(--ink);
}

/* Main title */
.ris-header {
    border-bottom: 3px solid var(--ink);
    padding-bottom: 1rem;
    margin-bottom: 2rem;
}
.ris-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    letter-spacing: -0.02em;
    line-height: 1;
    color: var(--ink);
    margin: 0;
}
.ris-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--slate);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* Status pill */
.status-pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.status-running { background: #fef3c7; color: #92400e; border: 1px solid #f59e0b; }
.status-done    { background: #d1fae5; color: #065f46; border: 1px solid #10b981; }
.status-error   { background: #fee2e2; color: #991b1b; border: 1px solid #ef4444; }
.status-idle    { background: var(--mist); color: var(--slate); border: 1px solid #c4bdb2; }

/* Audit trail */
.audit-step {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
    padding: 0.6rem 0.8rem;
    background: var(--card-bg);
    border-left: 3px solid var(--amber);
    font-size: 0.84rem;
    color: var(--ink);
}
.audit-step .step-time {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--snippet-color);
    white-space: nowrap;
    min-width: 70px;
}
.audit-step .step-agent {
    font-weight: 600;
    color: var(--amber);
    min-width: 90px;
}
.audit-step.step-error { border-left-color: var(--rust); }
.audit-step.step-error .step-agent { color: var(--rust); }
.audit-step.step-success { border-left-color: var(--sage); }
.audit-step.step-success .step-agent { color: var(--sage); }

/* Warning box */
.redteam-warn {
    background: var(--warn-bg);
    border: 1.5px solid var(--amber);
    border-left: 5px solid var(--rust);
    padding: 1rem;
    margin: 0.5rem 0;
    font-size: 0.87rem;
    color: var(--ink);
}
.redteam-warn strong { color: var(--rust); }

/* Evidence table */
.evidence-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; color: var(--ink); }
.evidence-table th {
    background: var(--ink);
    color: var(--paper);
    padding: 0.5rem 0.75rem;
    text-align: left;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.evidence-table td { padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--mist); vertical-align: top; color: var(--ink); }
.evidence-table tr:nth-child(even) td { background: var(--even-row-bg); }
.conf-high { color: var(--sage); font-weight: 600; }
.conf-med  { color: var(--amber); font-weight: 600; }
.conf-low  { color: var(--rust);  font-weight: 600; }

/* Section headers */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--slate);
    border-bottom: 1px solid var(--mist);
    padding-bottom: 0.3rem;
    margin-bottom: 0.75rem;
}

/* Path card */
.path-card {
    border: 1.5px solid var(--card-border);
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    background: var(--card-bg);
    color: var(--ink);
}
.path-card strong { color: var(--slate); }

/* Confidence badge */
.conf-badge {
    display: inline-block;
    padding: 0.1rem 0.45rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    border-radius: 2px;
}

/* Streamlit overrides */
div[data-testid="stSidebar"] { background: var(--ink) !important; }
div[data-testid="stSidebar"] * { color: var(--paper) !important; }
div[data-testid="stSidebar"] .stTextInput input { background: #1a1a1a !important; border-color: #444 !important; color: white !important; }
div[data-testid="stSidebar"] label { color: #aaa !important; }

.stButton > button {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.5rem !important;
}
.stButton > button:hover {
    background: var(--amber) !important;
    color: var(--ink) !important;
}

.stTextArea textarea {
    border: 1.5px solid var(--mist) !important;
    border-radius: 2px !important;
    font-family: 'DM Sans', sans-serif !important;
    background: var(--card-bg) !important;
    color: var(--ink) !important;
}
.stTextArea textarea:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 1px var(--amber) !important;
}

.stRadio > div { gap: 0.4rem; }

.stProgress > div > div { background: var(--amber) !important; }

hr { border-color: var(--mist) !important; }

/* Pipeline connector lines */
.pipe-connector { background: var(--mist) !important; }

/* Logical gap items */
.gap-item {
    font-size: 0.8rem;
    padding: 0.25rem 0.5rem;
    border-left: 2px solid var(--amber);
    margin-bottom: 0.25rem;
    color: var(--ink);
    background: var(--card-bg);
}

/* RAG chunk card */
.rag-chunk {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-left: 3px solid var(--sage);
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
    color: var(--ink);
}
.rag-chunk .rag-source {
    font-family: 'DM Mono', monospace;
    font-size: 0.66rem;
    color: var(--sage);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem;
}

/* History item */
.hist-item {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.35rem;
    font-size: 0.82rem;
    cursor: pointer;
    color: var(--ink);
    transition: border-color 0.15s;
}
.hist-item:hover { border-color: var(--amber); }
.hist-item .hist-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--snippet-color);
    margin-top: 0.2rem;
}

/* Contradiction card */
.contra-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-left: 4px solid var(--rust);
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.83rem;
    color: var(--ink);
}
.contra-card .contra-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--rust);
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}

/* Gap-fill banner */
.gapfill-banner {
    background: var(--card-bg);
    border: 1px dashed var(--amber);
    padding: 0.5rem 0.8rem;
    font-size: 0.8rem;
    color: var(--amber);
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.5rem;
}

/* Footer */
.ris-footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--mist);
    text-align: center;
    font-family: monospace;
    font-size: 0.68rem;
    color: var(--snippet-color);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ─── LLM Client ───────────────────────────────────────────────────────────────
def call_llm(
    messages: List[Dict],
    system: str,
    api_key: str,
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> str:
    """Call OpenRouter LLM."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://research-assistant.app",
        "X-Title": "Research Intelligence System",
    }
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()
    # Track token usage
    usage = result.get("usage", {})
    total_tok = usage.get("total_tokens", estimate_tokens(str(messages) + system))
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    st.session_state.token_count += total_tok
    return result["choices"][0]["message"]["content"]


def call_llm_json(messages, system, api_key, model, **kw) -> Dict:
    """Call LLM and parse JSON, stripping markdown fences."""
    raw = call_llm(messages, system, api_key, model, **kw)
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```", 2)[-1] if clean.count("```") >= 2 else clean
        if clean.startswith("json"):
            clean = clean[4:]
        if "```" in clean:
            clean = clean[: clean.rfind("```")]
    try:
        return json.loads(clean.strip())
    except json.JSONDecodeError:
        # Fallback: extract first {...} block
        import re
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            return json.loads(m.group())
        return {"raw": raw, "parse_error": True}


# ─── PDF RAG Engine ───────────────────────────────────────────────────────────
def extract_pdf_text(pdf_bytes: bytes) -> List[Dict]:
    """Extract text chunks from a PDF file."""
    if not HAS_RAG:
        return []
    try:
        import io as _io
        reader = PyPDF2.PdfReader(_io.BytesIO(pdf_bytes))
        chunks = []
        for pg_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Split into ~400-char chunks with 50-char overlap
            stride, size = 350, 400
            for i in range(0, max(1, len(text)), stride):
                chunk = text[i:i+size].strip()
                if len(chunk) > 60:
                    chunks.append({"page": pg_num+1, "text": chunk, "char_offset": i})
        return chunks
    except Exception:
        return []


def simple_embed(text: str) -> "np.ndarray":
    """Lightweight bag-of-words embedding (no external model needed)."""
    import hashlib as _hl, numpy as _np, re as _re
    words = _re.findall(r"[a-z]{3,}", text.lower())
    vec = _np.zeros(512, dtype=_np.float32)
    for w in words:
        idx = int(_hl.md5(w.encode()).hexdigest()[:4], 16) % 512
        vec[idx] += 1.0
    norm = _np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def build_rag_index(chunks: List[Dict]) -> tuple:
    """Build a FAISS flat index from chunks."""
    if not HAS_RAG or not chunks:
        return None, []
    vecs = np.stack([simple_embed(c["text"]) for c in chunks])
    index = faiss.IndexFlatIP(512)
    faiss.normalize_L2(vecs)
    index.add(vecs)
    return index, chunks


def rag_search(query: str, index, chunks: List[Dict], k: int = 5) -> List[Dict]:
    """Search the FAISS index for top-k chunks."""
    if index is None or not chunks:
        return []
    q_vec = simple_embed(query).reshape(1, -1)
    faiss.normalize_L2(q_vec)
    scores, idxs = index.search(q_vec, min(k, len(chunks)))
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx >= 0 and score > 0.05:
            c = chunks[idx].copy()
            c["score"] = float(score)
            results.append(c)
    return results



def tool_wikipedia(query: str, sentences: int = 5) -> Dict:
    """Wikipedia summary retrieval."""
    if not HAS_WIKI:
        return {"source": "wikipedia", "error": "wikipedia package not installed", "content": ""}
    try:
        result = wikipedia.summary(query, sentences=sentences, auto_suggest=True)
        page = wikipedia.page(query, auto_suggest=True)
        return {
            "source": "wikipedia",
            "title": page.title,
            "url": page.url,
            "content": result,
            "query": query,
        }
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            result = wikipedia.summary(e.options[0], sentences=sentences)
            page = wikipedia.page(e.options[0])
            return {
                "source": "wikipedia",
                "title": page.title,
                "url": page.url,
                "content": result,
                "query": query,
            }
        except Exception:
            return {"source": "wikipedia", "error": str(e), "content": "", "query": query}
    except Exception as e:
        return {"source": "wikipedia", "error": str(e), "content": "", "query": query}


def tool_arxiv(query: str, max_results: int = 3) -> List[Dict]:
    """ArXiv paper search."""
    if not HAS_ARXIV:
        return [{"source": "arxiv", "error": "arxiv package not installed"}]
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in client.results(search):
            summary_text = (paper.summary or "").strip()
            results.append({
                "source": "arxiv",
                "title": paper.title,
                "authors": [a.name for a in paper.authors[:3]],
                "summary": summary_text[:600],
                "content": summary_text[:600],
                "url": paper.entry_id,
                "published": str(paper.published.date()),
                "query": query,
            })
        return results if results else [{"source": "arxiv", "content": "No papers found", "query": query}]
    except Exception as e:
        return [{"source": "arxiv", "error": str(e), "query": query}]


def tool_tavily(query: str, api_key: str, max_results: int = 4) -> List[Dict]:
    """Tavily web search."""
    if not api_key:
        return [{"source": "tavily", "error": "No Tavily API key", "query": query}]
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "advanced",
                "include_answer": True,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        if data.get("answer"):
            results.append({
                "source": "tavily_answer",
                "content": data["answer"],
                "query": query,
                "url": "tavily-synthesis",
            })
        for r in data.get("results", []):
            results.append({
                "source": "tavily",
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:800],
                "score": r.get("score", 0),
                "query": query,
            })
        return results
    except Exception as e:
        return [{"source": "tavily", "error": str(e), "query": query}]


def run_tools_parallel(queries: List[Dict], tavily_key: str) -> List[Dict]:
    """Execute tool calls in parallel."""
    results = []

    def fetch_one(item):
        tool = item.get("tool", "wikipedia")
        q = item.get("query", "")
        try:
            if tool == "wikipedia":
                return tool_wikipedia(q)
            elif tool == "arxiv":
                return tool_arxiv(q)
            elif tool == "tavily":
                return tool_tavily(q, tavily_key)
            else:
                return tool_wikipedia(q)
        except Exception as e:
            return {"source": tool, "error": str(e), "query": q}

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_one, item): item for item in queries}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                if isinstance(r, list):
                    results.extend(r)
                else:
                    results.append(r)
            except Exception as e:
                results.append({"error": str(e)})
    return results


# ─── Token Estimator ──────────────────────────────────────────────────────────
# Approximate: 1 token ≈ 4 chars (good enough without tiktoken)
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

# Rough cost table per 1K tokens (input+output blended), USD
TOKEN_COST = {
    "openai/gpt-4o-mini":           0.00030,
    "openai/gpt-4o":                0.00750,
    "anthropic/claude-3.5-haiku":   0.00040,
    "anthropic/claude-3.5-sonnet":  0.00450,
    "google/gemini-flash-1.5":      0.00015,
    "mistralai/mistral-nemo":       0.00015,
}


def log_step(state: ResearchState, agent: str, msg: str, kind: str = "info") -> ResearchState:
    trail = list(state.get("audit_trail", []))
    trail.append({
        "ts": datetime.now().strftime("%H:%M:%S"),
        "agent": agent,
        "message": msg,
        "kind": kind,
    })
    return {**state, "audit_trail": trail}


# ─── Agent Nodes ──────────────────────────────────────────────────────────────
def planner_agent(state: ResearchState, cfg: Dict) -> ResearchState:
    state = log_step(state, "PLANNER", f"Decomposing query: '{state['query']}'")

    system = """You are a research planner. Decompose the user query into structured sub-questions
and 3 distinct investigation paths (angles). Respond ONLY with valid JSON (no markdown, no preamble).
Schema:
{
  "sub_questions": [{"id":"sq1","question":"...","priority":"high|med|low"},...],
  "investigation_paths": [
    {"id":"path_a","name":"...","description":"...","focus":"...","tools":["wikipedia","arxiv","tavily"]},
    {"id":"path_b","name":"...","description":"...","focus":"...","tools":[...]},
    {"id":"path_c","name":"...","description":"...","focus":"...","tools":[...]}
  ],
  "ambiguity_notes": "..."
}"""
    try:
        result = call_llm_json(
            [{"role": "user", "content": state["query"]}],
            system,
            cfg["openrouter_key"],
            cfg["model"],
            max_tokens=1800,
        )
        if result.get("parse_error"):
            raise ValueError("JSON parse failed")
        state = log_step(state, "PLANNER", f"Generated {len(result.get('sub_questions',[]))} sub-questions and {len(result.get('investigation_paths',[]))} paths", "success")
        return {
            **state,
            "sub_questions": result.get("sub_questions", []),
            "investigation_paths": result.get("investigation_paths", []),
            "status": "paths_ready",
        }
    except Exception as e:
        state = log_step(state, "PLANNER", f"Error: {e}", "error")
        return {
            **state,
            "status": "planning_failed",
            "error": str(e),
        }


def retriever_agent(state: ResearchState, cfg: Dict) -> ResearchState:
    path_id = state.get("selected_path", "path_a")
    paths = state.get("investigation_paths", [])
    chosen = next((p for p in paths if p["id"] == path_id), paths[0] if paths else {})
    tools = chosen.get("tools", ["wikipedia", "tavily"])
    if HAS_ARXIV and "arxiv" not in tools:
        tools.append("arxiv")

    state = log_step(state, "RETRIEVER", f"Retrieving via path '{chosen.get('name','?')}' using tools: {tools}")

    queries = []
    for sq in state.get("sub_questions", []):
        q = sq.get("question", state["query"])
        for t in tools:
            queries.append({"tool": t, "query": q})

    # Also query the main topic
    for t in tools:
        queries.append({"tool": t, "query": chosen.get("focus", state["query"])})

    # Guarantee ArXiv is one of the retrieval sources when available
    if HAS_ARXIV:
        main_topic = chosen.get("focus", state["query"])
        queries.append({"tool": "arxiv", "query": main_topic})

    results = run_tools_parallel(queries, cfg.get("tavily_key", ""))
    good = [r for r in results if not r.get("error") and r.get("content", "")]
    errs = [r for r in results if r.get("error")]

    state = log_step(state, "RETRIEVER", f"Retrieved {len(good)} results ({len(errs)} errors)", "success" if good else "error")
    if errs:
        for e in errs[:2]:
            state = log_step(state, "RETRIEVER", f"Tool error [{e.get('source','')}]: {e.get('error','?')}", "error")

    # PDF RAG retrieval (if index is loaded in session state)
    rag_hits = []
    if st.session_state.get("rag_index") is not None:
        idx   = st.session_state["rag_index"]
        cks   = st.session_state["rag_chunks"]
        # Search for each decomposed sub-question
        for sq in state.get("sub_questions", []):
            hits = rag_search(sq.get("question", state["query"]), idx, cks, k=3)
            for h in hits:
                h["source"] = "pdf_rag"
                h["content"] = h.get("text", "")
                h["title"] = f"PDF p.{h.get('page','?')}"
            rag_hits.extend(hits)

        # Also search the path focus (main topic) so RAG is included even when sub-questions are scarce
        focus_q = chosen.get("focus", state.get("query", ""))
        if focus_q:
            focus_hits = rag_search(focus_q, idx, cks, k=4)
            for h in focus_hits:
                h["source"] = "pdf_rag"
                h["content"] = h.get("text", "")
                h["title"] = f"PDF p.{h.get('page','?')}"
            # avoid simple duplicates by extending; analysis_agent will handle corpus merging
            rag_hits.extend([h for h in focus_hits if h not in rag_hits])
        if rag_hits:
            state = log_step(state, "RETRIEVER", f"PDF RAG: {len(rag_hits)} chunks retrieved", "success")
            # expose last rag hits to the UI session for immediate visibility
            st.session_state.last_rag_hits = rag_hits

    # Merge PDF hits into the main retrieval_results so they appear in evidence and reporting
    merged_results = list(results) + [
        {**h, "url": "", "title": h.get("title", "PDF chunk"), "source": "pdf_rag"}
        for h in rag_hits
    ]

    return {**state, "retrieval_results": merged_results, "rag_results": rag_hits, "status": "retrieved"}


def analysis_agent(state: ResearchState, cfg: Dict) -> ResearchState:
    state = log_step(state, "ANALYST", "Running critical analysis and contradiction detection")

    results = state.get("retrieval_results", [])
    good = [r for r in results if r.get("content") and not r.get("error")]
    if not good:
        state = log_step(state, "ANALYST", "No usable retrieval results", "error")
        return {**state, "analysis": {"summary": "No data retrieved.", "contradictions": [], "confidence": 0.2}, "status": "analyzed"}

    corpus = "\n\n".join(
        f"[{r.get('source','?')}] {r.get('title','')} — {r.get('content','')[:600]}"
        for r in good[:10]
    )
    # Append PDF RAG hits
    rag_good = [r for r in state.get("rag_results", []) if r.get("content")]
    if rag_good:
        corpus += "\n\n" + "\n\n".join(
            f"[PDF p.{r.get('page','?')}] {r.get('content','')[:400]}"
            for r in rag_good[:6]
        )

    system = """You are a critical research analyst. Analyze the corpus for the given query.
Respond ONLY with valid JSON:
{
  "summary": "concise 3-5 sentence summary",
  "key_findings": ["finding 1", "finding 2", ...],
  "contradictions": [{"claim_a":"...","claim_b":"...","sources":"..."}],
  "data_gaps": ["gap 1", ...],
  "confidence_score": 0.0-1.0,
  "source_quality": "high|medium|low"
}"""
    try:
        result = call_llm_json(
            [{"role": "user", "content": f"Query: {state['query']}\n\nCorpus:\n{corpus}"}],
            system,
            cfg["openrouter_key"],
            cfg["model"],
            max_tokens=1800,
        )
        state = log_step(state, "ANALYST", f"Identified {len(result.get('contradictions',[]))} contradictions, confidence {result.get('confidence_score',0):.0%}", "success")
        return {**state, "analysis": result, "status": "analyzed"}
    except Exception as e:
        state = log_step(state, "ANALYST", f"Error: {e}", "error")
        return {**state, "analysis": {"summary": "Analysis failed.", "key_findings": [], "contradictions": [], "confidence_score": 0.3}, "status": "analyzed"}


def insight_agent(state: ResearchState, cfg: Dict) -> ResearchState:
    state = log_step(state, "INSIGHT", "Generating trends, hypotheses, and causal links")

    analysis = state.get("analysis", {})
    corpus_sample = "\n".join(
        r.get("content", "")[:300]
        for r in state.get("retrieval_results", [])[:6]
        if r.get("content") and not r.get("error")
    )

    system = """You are a research insight generator. Based on the analysis, generate actionable insights.
Respond ONLY with valid JSON:
{
  "trends": ["trend 1", ...],
  "opportunities": ["opportunity 1", ...],
  "risks": ["risk 1", ...],
  "hypotheses": ["hypothesis 1", ...],
  "causal_links": [{"cause":"...","effect":"...","confidence":"high|med|low"}],
  "strategic_implications": "paragraph text"
}"""
    try:
        prompt = f"Query: {state['query']}\nSummary: {analysis.get('summary','')}\nKey Findings: {analysis.get('key_findings',[])}\nCorpus sample: {corpus_sample}"
        result = call_llm_json(
            [{"role": "user", "content": prompt}],
            system,
            cfg["openrouter_key"],
            cfg["model"],
            max_tokens=1800,
        )
        state = log_step(state, "INSIGHT", f"Generated {len(result.get('trends',[]))} trends, {len(result.get('opportunities',[]))} opportunities", "success")
        return {**state, "insights": result, "status": "insights_ready"}
    except Exception as e:
        state = log_step(state, "INSIGHT", f"Error: {e}", "error")
        return {**state, "insights": {"trends": [], "opportunities": ["Further research needed"], "risks": [], "hypotheses": []}, "status": "insights_ready"}


def red_team_agent(state: ResearchState, cfg: Dict) -> ResearchState:
    state = log_step(state, "RED TEAM", "Validating for hallucinations, bias, logical gaps")

    analysis = state.get("analysis", {})
    insights = state.get("insights", {})

    system = """You are a red-team validator. Critically assess research for hallucinations, unsupported claims, bias, and logical gaps.
Respond ONLY with valid JSON:
{
  "hallucination_flags": [{"claim":"...","issue":"...","severity":"high|med|low"}],
  "bias_flags": [{"type":"...","description":"..."}],
  "logical_gaps": ["gap 1", ...],
  "unsupported_claims": ["claim 1", ...],
  "overall_reliability": "high|medium|low",
  "confidence_adjustment": -0.2 to 0.1,
  "validator_notes": "overall assessment paragraph"
}"""
    try:
        prompt = f"""Query: {state['query']}
Summary: {analysis.get('summary','')}
Key Findings: {json.dumps(analysis.get('key_findings',[]))}
Insights Trends: {json.dumps(insights.get('trends',[]))}
Opportunities: {json.dumps(insights.get('opportunities',[]))}
Contradictions found: {json.dumps(analysis.get('contradictions',[]))}"""
        result = call_llm_json(
            [{"role": "user", "content": prompt}],
            system,
            cfg["openrouter_key"],
            cfg["model"],
            max_tokens=1500,
        )
        n_flags = len(result.get("hallucination_flags", [])) + len(result.get("bias_flags", []))
        state = log_step(state, "RED TEAM", f"Found {n_flags} flags. Reliability: {result.get('overall_reliability','?')}", "success")
        return {**state, "red_team": result, "status": "validated"}
    except Exception as e:
        state = log_step(state, "RED TEAM", f"Error: {e}", "error")
        return {**state, "red_team": {"hallucination_flags": [], "bias_flags": [], "overall_reliability": "medium"}, "status": "validated"}


def gap_fill_agent(state: ResearchState, cfg: Dict) -> ResearchState:
    """Iteratively retrieves information to fill identified data gaps."""
    gaps = state.get("analysis", {}).get("data_gaps", [])
    rt_gaps = state.get("red_team", {}).get("logical_gaps", [])
    all_gaps = (gaps + rt_gaps)[:3]  # cap at 3 to avoid runaway cost

    if not all_gaps:
        state = log_step(state, "GAP-FILL", "No gaps to fill — skipping", "success")
        return {**state, "gap_fill_results": [], "status": "gap_filled"}

    state = log_step(state, "GAP-FILL", f"Filling {len(all_gaps)} identified gaps")
    extra_queries = [{"tool": "tavily", "query": g} for g in all_gaps]
    extra_queries += [{"tool": "wikipedia", "query": g} for g in all_gaps[:2]]

    fill_results = run_tools_parallel(extra_queries, cfg.get("tavily_key", ""))
    good = [r for r in fill_results if r.get("content") and not r.get("error")]
    state = log_step(state, "GAP-FILL", f"Retrieved {len(good)} gap-filling results", "success")

    # Merge gap-fill results back into retrieval_results
    merged = list(state.get("retrieval_results", [])) + good
    return {**state, "retrieval_results": merged, "gap_fill_results": good, "status": "gap_filled"}


def report_agent(state: ResearchState, cfg: Dict) -> ResearchState:
    state = log_step(state, "REPORTER", "Compiling final structured research report")

    analysis  = state.get("analysis", {})
    insights  = state.get("insights", {})
    red_team  = state.get("red_team", {})
    results   = state.get("retrieval_results", [])
    conf_base = float(analysis.get("confidence_score", 0.5))
    conf_adj  = float(red_team.get("confidence_adjustment", 0))
    final_conf = max(0.0, min(1.0, conf_base + conf_adj))

    # Build evidence table — prioritize PDF sources first
    evidence = []
    seen_urls = set()
    # Separate PDF and non-PDF results
    pdf_results = [r for r in results if r.get("source") == "pdf_rag" and r.get("content")]
    other_results = [r for r in results if r.get("source") != "pdf_rag" and r.get("content") and not r.get("error")]
    # Process PDF results first
    for r in pdf_results[:6]:  # Limit PDF to top 6
        url = r.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        evidence.append({
            "source": r.get("source", "?"),
            "title": r.get("title", r.get("query", ""))[:70],
            "url": url,
            "snippet": r.get("content", "")[:200],
            "confidence": "high" if r.get("score", 0.8) > 0.7 else "medium",
        })
    # Then add other results
    for r in other_results[:6]:  # Limit others to top 6, total up to 12
        url = r.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        evidence.append({
            "source": r.get("source", "?"),
            "title": r.get("title", r.get("query", ""))[:70],
            "url": url,
            "snippet": r.get("content", "")[:200],
            "confidence": "high" if r.get("score", 0.8) > 0.7 else "medium",
        })

    report = {
        "title": f"Research Report: {state['query'][:80]}",
        "generated_at": datetime.now().isoformat(),
        "query": state["query"],
        "executive_summary": analysis.get("summary", "No summary available."),
        "key_findings": analysis.get("key_findings", []),
        "insights": {
            "trends": insights.get("trends", []),
            "opportunities": insights.get("opportunities", []),
            "risks": insights.get("risks", []),
            "hypotheses": insights.get("hypotheses", []),
            "strategic_implications": insights.get("strategic_implications", ""),
        },
        "evidence_table": evidence[:12],
        "contradictions": analysis.get("contradictions", []),
        "data_gaps": analysis.get("data_gaps", []),
        "red_team_summary": {
            "overall_reliability": red_team.get("overall_reliability", "unknown"),
            "hallucination_flags": red_team.get("hallucination_flags", []),
            "bias_flags": red_team.get("bias_flags", []),
            "logical_gaps": red_team.get("logical_gaps", []),
            "validator_notes": red_team.get("validator_notes", ""),
        },
        "confidence_score": final_conf,
        "confidence_explanation": "",
        "selected_path": state.get("selected_path", ""),
        "selected_path_name": "",  # Will be populated below
        "investigation_paths": state.get("investigation_paths", []),
        "gap_fill_count": len(state.get("gap_fill_results", [])),
        "rag_chunks_used": len([r for r in state.get("rag_results", []) if r.get("content")]),
    }
    
    # Look up and add the path name
    path_id = state.get("selected_path", "")
    for p in state.get("investigation_paths", []):
        if p.get("id") == path_id:
            path_name = p.get("name", "")
            # Truncate path name if too long (max 50 chars)
            if len(path_name) > 50:
                path_name = path_name[:47] + "..."
            report["selected_path_name"] = path_name
            break
    
    # Generate a concise 1-liner title from the query using LLM
    try:
        title_prompt = [{"role": "user", "content": f"Summarize this research query in a single sentence (max 100 chars), suitable as a report title:\n\n{state['query']}"}]
        title_system = "You are a concise summarizer. Respond with ONLY a single sentence title, no quotes or preamble."
        query_title = call_llm(title_prompt, title_system, cfg["openrouter_key"], cfg["model"], max_tokens=50)
        query_title = query_title.strip().strip('"').strip("'")[:100]
    except Exception:
        # Fallback: truncate query
        query = state["query"].strip()
        query_title = query[:97] + "..." if len(query) > 100 else query
    
    report["title"] = f"Research Report: {query_title}"
    # Build a concise explanation of how confidence was calculated
    try:
        evidence_count = len(evidence)
    except Exception:
        evidence_count = len([r for r in results if r.get("content") and not r.get("error")])
    rag_used = len([r for r in state.get("rag_results", []) if r.get("content")])
    explanation = (
        f"Calculated as analysis confidence ({conf_base:.0%}) adjusted by red-team ({conf_adj:+.0%}) = {final_conf:.0%}. "
        f"Analysis confidence is the LLM's assessment based on the retrieved corpus; red-team adjustment reflects validator findings. "
        f"Evidence sources: {evidence_count}; PDF chunks used: {rag_used}."
    )
    report["confidence_explanation"] = explanation

    state = log_step(state, "REPORTER", f"Report compiled. Final confidence: {final_conf:.0%}", "success")
    return {**state, "final_report": report, "status": "complete"}


# ─── LangGraph Builder ────────────────────────────────────────────────────────
def build_graph(cfg: Dict):
    """Build the LangGraph state machine."""
    from langgraph.graph import StateGraph, END

    def wrap(fn):
        def node(state):
            return fn(state, cfg)
        return node

    g = StateGraph(ResearchState)
    g.add_node("planner",   wrap(planner_agent))
    g.add_node("retriever", wrap(retriever_agent))
    g.add_node("analyst",   wrap(analysis_agent))
    g.add_node("insight",   wrap(insight_agent))
    g.add_node("red_team",  wrap(red_team_agent))
    g.add_node("gap_fill",  wrap(gap_fill_agent))
    g.add_node("reporter",  wrap(report_agent))

    g.set_entry_point("planner")
    g.add_edge("planner",   "retriever")
    g.add_edge("retriever", "analyst")
    g.add_edge("analyst",   "insight")
    g.add_edge("insight",   "red_team")
    g.add_edge("red_team",  "gap_fill")
    g.add_edge("gap_fill",  "reporter")
    g.add_edge("reporter",  END)

    return g.compile()


# ─── Session state init ───────────────────────────────────────────────────────
for key, default in [
    ("phase", "idle"),
    ("planned_state", None),
    ("final_state", None),
    ("audit_trail", []),
    ("token_count", 0),
    ("query_history", []),   # list of {query, ts, conf, summary}
    ("rag_index", None),
    ("rag_chunks", []),
    ("rag_filename", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─── UI Layout ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ris-header">
  <div class="ris-title">Research Intelligence System</div>
  <div class="ris-subtitle">Multi-Agent · LangGraph Orchestration · Transparent Reasoning</div>
</div>
""", unsafe_allow_html=True)

# Sidebar config
with st.sidebar:
    st.markdown("### ⚙ Configuration")
    openrouter_key = st.text_input("OpenRouter API Key", type="password", placeholder="sk-or-...")
    tavily_key     = st.text_input("Tavily API Key (optional)", type="password", placeholder="tvly-...")
    model          = st.selectbox("Model", [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "anthropic/claude-3.5-haiku",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-flash-1.5",
        "mistralai/mistral-nemo",
    ])
    st.markdown("---")
    st.markdown("**Tools Available**")
    st.markdown(f"- Wikipedia: {'✅' if HAS_WIKI else '❌ (install `wikipedia`)'}")
    st.markdown(f"- ArXiv: {'✅' if HAS_ARXIV else '❌ (install `arxiv`)'}")
    st.markdown(f"- Tavily: {'✅' if tavily_key else '⚠ (no key)'}")
    st.markdown("---")
    st.markdown("**System**")
    st.markdown(f"- LangGraph: {'✅' if HAS_LANGGRAPH else '❌'}")
    st.markdown("- Agents: 6 specialized")
    st.markdown("- Parallel retrieval: ✅")
    st.markdown("---")
    st.markdown("**PDF RAG Source (optional)**")
    if not HAS_RAG:
        st.info("PDF RAG requires `faiss`, `PyPDF2`, and `numpy`. Install to enable PDF ingestion.")
    else:
        pdf_file = st.file_uploader("Upload PDF to include as RAG source", type=["pdf"] )
        if pdf_file is not None:
            try:
                pdf_bytes = pdf_file.read()
                with st.spinner("Extracting PDF text and building RAG index..."):
                    chunks = extract_pdf_text(pdf_bytes)
                    if chunks:
                        idx, chunks_list = build_rag_index(chunks)
                        st.session_state.rag_index = idx
                        st.session_state.rag_chunks = chunks_list
                        st.session_state.rag_filename = getattr(pdf_file, "name", "uploaded.pdf")
                        st.success(f"Loaded {st.session_state.rag_filename} — {len(chunks_list)} chunks")
                    else:
                        st.warning("No extractable text found in PDF.")
            except Exception as e:
                st.warning(f"Failed to process PDF: {e}")

        # Show loaded PDF info and allow clearing
        if st.session_state.get("rag_filename"):
            st.markdown(f"- Loaded PDF: {st.session_state.get('rag_filename')} ({len(st.session_state.get('rag_chunks', []))} chunks)")
            if st.button("Clear uploaded PDF"):
                st.session_state.rag_index = None
                st.session_state.rag_chunks = []
                st.session_state.rag_filename = ""

cfg = {"openrouter_key": openrouter_key, "tavily_key": tavily_key, "model": model}

# Main columns
left, right = st.columns([3, 2])

with left:
    # Query input
    st.markdown('<div class="section-label">Research Query</div>', unsafe_allow_html=True)
    query = st.text_area(
        "Enter your research question",
        placeholder='e.g. "Analyze the competitive landscape of AI coding assistants and identify market opportunities"',
        height=100,
        label_visibility="collapsed",
    )

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        # Show Plan Research only when not yet planned (or reset after complete)
        plan_active = st.session_state.phase in ("idle", "complete", "planning_failed")
        plan_btn = st.button(
            "🔍 Plan Research",
            disabled=not (query and openrouter_key) or not plan_active,
            key="plan_btn",
        )
    with col_btn2:
        # Show Run Analysis only after path is selected; disabled otherwise
        planned_state = st.session_state.get("planned_state")
        run_active = (st.session_state.phase in ("planned", "path_selected") and 
                      planned_state and planned_state.get("status") == "paths_ready")
        run_btn = st.button(
            "▶ Run Analysis",
            disabled=not run_active,
            key="run_btn",
        )

    # ── Phase 1: Planning ─────────────────────────────────────────────────────
    if plan_btn and query and openrouter_key:
        st.session_state.phase = "planning"
        st.session_state.final_state = None

        with st.spinner("Planner agent decomposing query..."):
            init_state = ResearchState(
                query=query,
                sub_questions=[],
                selected_path="path_a",
                investigation_paths=[],
                retrieval_results=[],
                rag_results=[],
                analysis={},
                insights={},
                red_team={},
                gap_fill_results=[],
                final_report={},
                audit_trail=[],
                status="init",
                error=None,
                iteration=0,
                token_count=0,
            )
            planned = planner_agent(init_state, cfg)
            st.session_state.planned_state = planned
            st.session_state.audit_trail = planned.get("audit_trail", [])
            if planned.get("status") == "paths_ready":
                st.session_state.phase = "path_selected"  # Automatically enable Run Analysis with default path
            else:
                st.session_state.phase = "planning_failed"

    # ── Planning failed ───────────────────────────────────────────────────────
    if st.session_state.phase == "planning_failed" and st.session_state.planned_state:
        st.info("Planning failed. Please check your OpenRouter API key and selected model, then try planning again.")

    # ── Path selection ────────────────────────────────────────────────────────
    if st.session_state.phase in ("planned", "path_selected") and st.session_state.planned_state:
        ps = st.session_state.planned_state
        paths = ps.get("investigation_paths", [])
        sqs   = ps.get("sub_questions", [])

        if sqs:
            st.markdown('<div class="section-label" style="margin-top:1.2rem">Decomposed Sub-Questions</div>', unsafe_allow_html=True)
            for sq in sqs:
                pri_color = {"high": "#b84c2e", "med": "#d4871a", "low": "#3d6b52"}.get(sq.get("priority","med"), "#888")
                st.markdown(
                    f'<div style="padding:0.4rem 0.7rem;margin-bottom:0.3rem;background:var(--card-bg);border-left:3px solid {pri_color};font-size:0.85rem;color:var(--ink)">'
                    f'<span style="color:{pri_color};font-weight:600;font-size:0.7rem;text-transform:uppercase;font-family:monospace">{sq.get("priority","?")}</span> &nbsp; {sq.get("question","")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if paths:
            st.markdown('<div class="section-label" style="margin-top:1.2rem">Select Investigation Path</div>', unsafe_allow_html=True)
            path_labels = [f"{p['name']} — {p['description'][:80]}" for p in paths]
            path_ids    = [p["id"] for p in paths]
            selected_label = st.radio(
                "Choose a research angle:",
                path_labels,
                label_visibility="collapsed",
            )
            sel_idx = path_labels.index(selected_label)
            selected_path_id = path_ids[sel_idx]

            chosen = paths[sel_idx]
            st.markdown(
                f'<div class="path-card"><strong>Focus:</strong> {chosen.get("focus","")}<br>'
                f'<strong>Tools:</strong> {", ".join(chosen.get("tools",[]))}</div>',
                unsafe_allow_html=True,
            )
            st.session_state.planned_state["selected_path"] = selected_path_id
            st.session_state.phase = "path_selected"

    # ── Phase 2: Full pipeline run ────────────────────────────────────────────
    if run_btn and st.session_state.planned_state:
        st.session_state.phase = "running"
        st.session_state.live_pipeline_html = None  # Clear any previous live pipeline
        planned = st.session_state.planned_state

        progress_bar = st.progress(0, text="Initializing pipeline...")
        status_placeholder = st.empty()

        agents = [
            ("retriever", retriever_agent, 0.2, "Retrieving from sources..."),
            ("analyst",   analysis_agent,  0.4, "Analyzing and detecting contradictions..."),
            ("insight",   insight_agent,   0.6, "Generating insights..."),
            ("red_team",  red_team_agent,  0.8, "Red-team validation..."),
            ("reporter",  report_agent,    1.0, "Compiling report..."),
        ]

        state = dict(planned)
        trail = list(state.get("audit_trail", []))

        def update_live_pipeline():
            """Update the live pipeline HTML in session state for RHS display."""
            agents_list = [
                ("🗺", "Planner",   "Query decomposition"),
                ("🔍", "Retriever", "Multi-tool retrieval"),
                ("🧠", "Analyst",   "Contradiction detection"),
                ("💡", "Insight",   "Trend & hypothesis"),
                ("🎯", "Red Team",  "Hallucination check"),
                ("📋", "Reporter",  "Structured output"),
            ]
            agent_statuses = {}
            for step in st.session_state.audit_trail:
                agent_statuses[step["agent"]] = step["kind"]
            
            # Set next agent to running
            agent_order = ["PLANNER", "RETRIEVER", "ANALYST", "INSIGHT", "RED TEAM", "REPORTER"]
            for ag in agent_order:
                if ag not in agent_statuses or agent_statuses[ag] not in ["success", "error"]:
                    agent_statuses[ag] = "running"
                    break
            
            pipeline_html = '<div style="margin-bottom:1rem">'
            for icon, name, desc in agents_list:
                key = name.upper().replace(" ", " ")
                status = agent_statuses.get(key, "pending")
                if status == "success":
                    dot = '●'; dot_color = "#3d6b52"; name_style = ""
                elif status == "error":
                    dot = '●'; dot_color = "#b84c2e"; name_style = ""
                elif status == "running":
                    dot = '●'; dot_color = "#d4871a"; name_style = 'style="color:#d4871a;font-weight:bold;"'
                else:
                    dot = '○'; dot_color = "#bbb"; name_style = ""
                pipeline_html += f'<div style="display:flex;align-items:center;gap:0.5rem;padding:0.35rem 0;font-size:0.82rem">'
                pipeline_html += f'<span style="color:{dot_color};font-size:0.9rem">{dot}</span>'
                pipeline_html += f'<span>{icon} <strong {name_style}>{name}</strong> <span style="color:#999;font-size:0.75rem">— {desc}</span></span></div>'
                if name != "Reporter":
                    pipeline_html += '<div style="margin-left:0.55rem;width:1px;height:8px;background:#ddd"></div>'
            pipeline_html += '</div>'
            st.session_state.live_pipeline_html = pipeline_html

        # Preference executing the compiled LangGraph when available; fall back to manual sequential agents
        if HAS_LANGGRAPH:
            try:
                compiled = build_graph(cfg)
                # Try common invocation styles for compiled graphs
                try:
                    state = compiled(state)
                except Exception:
                    try:
                        state = compiled.run(state)
                    except Exception:
                        try:
                            state = compiled.execute(state)
                        except Exception as e_call:
                            raise e_call

                # update audit trail and progress after graph-run
                trail = list(state.get("audit_trail", []))
                st.session_state.audit_trail = trail
                update_live_pipeline()  # Update for RHS
                progress_bar.progress(1.0, text="Complete (LangGraph)")
                status_placeholder.markdown(
                    '<span class="status-pill status-done">✓ Analysis Complete (LangGraph)</span>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                # record graph failure and fall back to manual execution
                trail.append({"ts": datetime.now().strftime("%H:%M:%S"), "agent": "GRAPH", "message": f"LangGraph run failed: {e}", "kind": "error"})
                st.session_state.audit_trail = trail
                # Fall back to manual sequential execution with progress updates
                for name, fn, prog, msg in agents:
                    progress_bar.progress(prog - 0.15, text=msg)
                    status_placeholder.markdown(
                        f'<span class="status-pill status-running">● {msg}</span>',
                        unsafe_allow_html=True,
                    )
                    try:
                        state = fn(state, cfg)
                        trail = list(state.get("audit_trail", []))
                        st.session_state.audit_trail = trail
                        update_live_pipeline()  # Update for RHS
                    except Exception as e2:
                        trail.append({"ts": datetime.now().strftime("%H:%M:%S"), "agent": name.upper(), "message": f"Fatal: {e2}", "kind": "error"})
                        st.session_state.audit_trail = trail
                        update_live_pipeline()  # Update for RHS

                    progress_bar.progress(prog, text=msg)
        else:
            # No LangGraph installed; run the agents sequentially (existing behaviour)
            for name, fn, prog, msg in agents:
                progress_bar.progress(prog - 0.15, text=msg)
                status_placeholder.markdown(
                    f'<span class="status-pill status-running">● {msg}</span>',
                    unsafe_allow_html=True,
                )
                try:
                    state = fn(state, cfg)
                    trail = list(state.get("audit_trail", []))
                    st.session_state.audit_trail = trail
                    update_live_pipeline()  # Update for RHS
                except Exception as e:
                    trail.append({"ts": datetime.now().strftime("%H:%M:%S"), "agent": name.upper(), "message": f"Fatal: {e}", "kind": "error"})
                    st.session_state.audit_trail = trail
                    update_live_pipeline()  # Update for RHS

                progress_bar.progress(prog, text=msg)

        progress_bar.progress(1.0, text="Complete!")
        status_placeholder.markdown(
            '<span class="status-pill status-done">✓ Analysis Complete</span>',
            unsafe_allow_html=True,
        )
        update_live_pipeline()  # Final update
        st.session_state.final_state = state
        st.session_state.phase = "complete"
        st.rerun()

    # ── Final report display ──────────────────────────────────────────────────
    if st.session_state.phase == "complete" and st.session_state.final_state:
        fs = st.session_state.final_state
        rpt = fs.get("final_report", {})
        rt  = rpt.get("red_team_summary", {})

        # Red team warnings
        h_flags = rt.get("hallucination_flags", [])
        b_flags = rt.get("bias_flags", [])
        if h_flags or b_flags:
            st.markdown('<div class="section-label" style="margin-top:1.5rem;color:#b84c2e">⚠ Red Team Warnings</div>', unsafe_allow_html=True)
            for f in h_flags:
                sev = f.get("severity","med")
                st.markdown(
                    f'<div class="redteam-warn"><strong>HALLUCINATION [{sev.upper()}]:</strong> {f.get("claim","")} — <em>{f.get("issue","")}</em></div>',
                    unsafe_allow_html=True,
                )
            for f in b_flags:
                st.markdown(
                    f'<div class="redteam-warn"><strong>BIAS [{f.get("type","?")}]:</strong> {f.get("description","")}</div>',
                    unsafe_allow_html=True,
                )

        # Confidence gauge (show numeric score and red-team qualitative label side-by-side)
        conf = rpt.get("confidence_score", 0.5)
        conf_color = "#3d6b52" if conf >= 0.7 else "#d4871a" if conf >= 0.4 else "#b84c2e"
        conf_label = "High" if conf >= 0.7 else "Medium" if conf >= 0.4 else "Low"
        # red-team qualitative label
        rt_label = rt.get("overall_reliability", "unknown") or "unknown"
        rt_l = str(rt_label).lower()
        rt_color = "#3d6b52" if rt_l == "high" else "#d4871a" if rt_l == "medium" else "#b84c2e"
        st.markdown(
            f'<div style="margin:1.2rem 0;padding:0.8rem 1rem;background:var(--card-bg);border:1.5px solid {conf_color};display:flex;align-items:center;gap:1rem;color:var(--ink)">'
            f'<div style="font-family:monospace;font-size:0.7rem;text-transform:uppercase;color:{conf_color};font-weight:600">Confidence</div>'
            f'<div style="flex:1;height:8px;background:var(--conf-bar-bg);border-radius:1px"><div style="width:{conf*100:.0f}%;height:100%;background:{conf_color}"></div></div>'
            f'<div style="display:flex;align-items:center;gap:0.6rem">'
            f'<div style="font-family:monospace;font-weight:700;color:{conf_color}">{conf:.0%} {conf_label}</div>'
            f'<div style="font-family:monospace;font-size:0.78rem;color:{rt_color};padding:0.2rem 0.5rem;border:1px solid {rt_color};border-radius:3px">Red-team: {str(rt_label).title()}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Show concise explanation for the confidence calculation
        conf_expl = rpt.get("confidence_explanation", "")
        if conf_expl:
            st.markdown(
                f'<div style="font-size:0.82rem;color:#666;margin-top:0.5rem">{conf_expl}</div>',
                unsafe_allow_html=True,
            )

        # Executive summary
        st.markdown('<div class="section-label" style="margin-top:1.5rem">Executive Summary</div>', unsafe_allow_html=True)
        st.markdown(rpt.get("executive_summary", ""))

        # PDF RAG Matches (visual confirmation) — show collapsed by default with a small preview
        rag_matches = fs.get("rag_results", []) or st.session_state.get("last_rag_hits", [])
        if rag_matches:
            st.markdown('<div class="section-label" style="margin-top:1.0rem">PDF Matches</div>', unsafe_allow_html=True)
            preview_count = 3
            # show a short inline preview of the first few chunks
            for i, m in enumerate(rag_matches[:preview_count], 1):
                score = f" ({m.get('score'):.2f})" if m.get('score') is not None else ""
                title = m.get('title', f"PDF p.{m.get('page','?')}")
                snippet = (m.get('content') or m.get('text',''))[:350]
                st.markdown(
                    f'<div class="rag-chunk"><div class="rag-source">PDF{score} — {title}</div>{snippet}...</div>',
                    unsafe_allow_html=True,
                )

            # if there are more, provide an expander to view the full list (collapsed by default)
            if len(rag_matches) > preview_count:
                with st.expander(f"Show all PDF Matches ({len(rag_matches)} chunks)", expanded=False):
                    for i, m in enumerate(rag_matches, 1):
                        score = f" ({m.get('score'):.2f})" if m.get('score') is not None else ""
                        title = m.get('title', f"PDF p.{m.get('page','?')}")
                        snippet = (m.get('content') or m.get('text',''))[:1000]
                        st.markdown(
                            f'<div class="rag-chunk"><div class="rag-source">PDF{score} — {title}</div>{snippet}...</div>',
                            unsafe_allow_html=True,
                        )

        # Key findings
        findings = rpt.get("key_findings", [])
        if findings:
            st.markdown('<div class="section-label" style="margin-top:1.2rem">Key Findings</div>', unsafe_allow_html=True)
            for i, f in enumerate(findings, 1):
                st.markdown(f'**{i}.** {f}')

        # Insights
        ins = rpt.get("insights", {})
        tab_trends, tab_opps, tab_risks = st.tabs(["📈 Trends", "💡 Opportunities", "⚠ Risks"])
        with tab_trends:
            for t in ins.get("trends", ["No trends identified"]):
                st.markdown(f"- {t}")
        with tab_opps:
            for o in ins.get("opportunities", ["No opportunities identified"]):
                st.markdown(f"- {o}")
        with tab_risks:
            for r in ins.get("risks", ["No risks identified"]):
                st.markdown(f"- {r}")

        if ins.get("strategic_implications"):
            st.markdown("**Strategic Implications:** " + ins["strategic_implications"])

        # Evidence table
        ev = rpt.get("evidence_table", [])
        if ev:
            st.markdown('<div class="section-label" style="margin-top:1.5rem">Evidence Table</div>', unsafe_allow_html=True)
            rows = ""
            for e in ev:
                c = e.get("confidence", "medium")
                c_class = {"high": "conf-high", "medium": "conf-med", "low": "conf-low"}.get(c, "conf-med")
                src_badge = f'<span style="font-family:monospace;font-size:0.68rem;background:var(--mist);color:var(--ink);padding:0.1rem 0.3rem">{e.get("source","?")}</span>'
                url = e.get("url", "")
                title = e.get("title", "")
                title_html = f'<a href="{url}" target="_blank" style="color:var(--ink)">{title}</a>' if url and url != "tavily-synthesis" else title
                rows += f"""<tr>
                  <td>{src_badge}</td>
                  <td style="font-size:0.8rem">{title_html}</td>
                  <td style="font-size:0.78rem;color:var(--snippet-color)">{e.get("snippet","")[:150]}…</td>
                  <td class="{c_class}">{c.upper()}</td>
                </tr>"""
            st.markdown(
                f'<table class="evidence-table"><thead><tr><th>Source</th><th>Title</th><th>Snippet</th><th>Confidence</th></tr></thead><tbody>{rows}</tbody></table>',
                unsafe_allow_html=True,
            )

        # ── Export as PDF ─────────────────────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top:1.5rem">Export</div>', unsafe_allow_html=True)

        try:
            import io
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                HRFlowable, KeepTogether,
            )
            from reportlab.lib.enums import TA_LEFT, TA_CENTER

            def build_pdf(rpt: Dict) -> bytes:
                buf = io.BytesIO()
                doc = SimpleDocTemplate(
                    buf,
                    pagesize=A4,
                    leftMargin=2*cm, rightMargin=2*cm,
                    topMargin=2*cm, bottomMargin=2*cm,
                    title=rpt.get("title", "Research Report"),
                )
                styles = getSampleStyleSheet()
                # Custom styles
                title_style = ParagraphStyle("RISTitle", parent=styles["Heading1"],
                    fontSize=20, spaceAfter=4, textColor=colors.HexColor("#0f0e0d"),
                    fontName="Helvetica-Bold")
                h2_style = ParagraphStyle("RISH2", parent=styles["Heading2"],
                    fontSize=12, spaceAfter=4, spaceBefore=14,
                    textColor=colors.HexColor("#0f0e0d"), fontName="Helvetica-Bold",
                    borderPad=2)
                label_style = ParagraphStyle("RISLabel", parent=styles["Normal"],
                    fontSize=7, textColor=colors.HexColor("#888888"),
                    fontName="Helvetica", spaceAfter=6,
                    textTransform="uppercase", letterSpacing=1.5)
                body_style = ParagraphStyle("RISBody", parent=styles["Normal"],
                    fontSize=9.5, leading=14, textColor=colors.HexColor("#1a1a1a"),
                    fontName="Helvetica", spaceAfter=6)
                bullet_style = ParagraphStyle("RISBullet", parent=body_style,
                    leftIndent=12, bulletIndent=0, spaceAfter=3)
                mono_style = ParagraphStyle("RISMono", parent=body_style,
                    fontName="Courier", fontSize=8.5, textColor=colors.HexColor("#555"),
                    backColor=colors.HexColor("#f5f3ef"), leftIndent=8, rightIndent=8)
                warn_style = ParagraphStyle("RISWarn", parent=body_style,
                    backColor=colors.HexColor("#fff3cd"),
                    textColor=colors.HexColor("#7d4a00"),
                    borderPad=4, leftIndent=8)

                conf = rpt.get("confidence_score", 0.5)
                conf_label = "High" if conf >= 0.7 else "Medium" if conf >= 0.4 else "Low"
                conf_hex = "#3d6b52" if conf >= 0.7 else "#d4871a" if conf >= 0.4 else "#b84c2e"

                story = []

                # Header
                story.append(Paragraph("Research Intelligence System", label_style))
                story.append(Paragraph(rpt.get("title", "Research Report"), title_style))
                story.append(Paragraph(
                    f"Generated: {rpt.get('generated_at','')[:19].replace('T',' ')} &nbsp;|&nbsp; "
                    f"Confidence: <font color='{conf_hex}'><b>{conf:.0%} {conf_label}</b></font> &nbsp;|&nbsp; "
                    f"Path: {rpt.get('selected_path_name', rpt.get('selected_path',''))}",
                    ParagraphStyle("meta", parent=body_style, fontSize=8, textColor=colors.HexColor("#666")),
                ))
                story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0f0e0d"), spaceAfter=12))

                # Original Query
                story.append(Paragraph("Research Query", h2_style))
                story.append(Paragraph(rpt.get("query", ""), body_style))
                story.append(Spacer(1, 8))

                # Executive Summary
                story.append(Paragraph("Executive Summary", h2_style))
                story.append(Paragraph(rpt.get("executive_summary", ""), body_style))
                story.append(Spacer(1, 8))

                # Key Findings
                findings = rpt.get("key_findings", [])
                if findings:
                    story.append(Paragraph("Key Findings", h2_style))
                    for i, f in enumerate(findings, 1):
                        story.append(Paragraph(f"<b>{i}.</b> {f}", bullet_style))
                    story.append(Spacer(1, 8))

                # Insights
                ins = rpt.get("insights", {})
                for section, key in [("Trends", "trends"), ("Opportunities", "opportunities"), ("Risks", "risks")]:
                    items = ins.get(key, [])
                    if items:
                        story.append(Paragraph(section, h2_style))
                        for item in items:
                            story.append(Paragraph(f"• {item}", bullet_style))
                        story.append(Spacer(1, 4))

                if ins.get("strategic_implications"):
                    story.append(Paragraph("Strategic Implications", h2_style))
                    story.append(Paragraph(ins["strategic_implications"], body_style))
                    story.append(Spacer(1, 8))

                # Red Team
                rt = rpt.get("red_team_summary", {})
                h_flags = rt.get("hallucination_flags", [])
                b_flags = rt.get("bias_flags", [])
                if h_flags or b_flags or rt.get("validator_notes"):
                    story.append(Paragraph("Red Team Validation", h2_style))
                    story.append(Paragraph(
                        f"Reliability: <b>{rt.get('overall_reliability','?').upper()}</b>",
                        body_style,
                    ))
                    for f in h_flags:
                        story.append(Paragraph(
                            f"⚠ HALLUCINATION [{f.get('severity','?').upper()}]: {f.get('claim','')} — {f.get('issue','')}",
                            warn_style,
                        ))
                    for f in b_flags:
                        story.append(Paragraph(
                            f"⚠ BIAS [{f.get('type','?')}]: {f.get('description','')}",
                            warn_style,
                        ))
                    if rt.get("validator_notes"):
                        story.append(Paragraph(rt["validator_notes"], mono_style))
                    story.append(Spacer(1, 8))

                # Evidence Table
                ev = rpt.get("evidence_table", [])
                if ev:
                    story.append(Paragraph("Evidence Table", h2_style))
                    tdata = [["Source", "Title", "Snippet", "Conf"]]
                    for e in ev[:10]:
                        tdata.append([
                            Paragraph(e.get("source","?"), ParagraphStyle("ts", parent=body_style, fontSize=7, fontName="Courier")),
                            Paragraph((e.get("title","") or "")[:60], ParagraphStyle("tt", parent=body_style, fontSize=8)),
                            Paragraph((e.get("snippet","") or "")[:120] + "…", ParagraphStyle("tc", parent=body_style, fontSize=7.5, textColor=colors.HexColor("#555"))),
                            Paragraph(e.get("confidence","med").upper(), ParagraphStyle("tcf", parent=body_style, fontSize=7.5, fontName="Helvetica-Bold",
                                textColor=colors.HexColor("#3d6b52") if e.get("confidence")=="high" else colors.HexColor("#d4871a"))),
                        ])
                    t = Table(tdata, colWidths=[2.5*cm, 4.5*cm, 8*cm, 2*cm], repeatRows=1)
                    t.setStyle(TableStyle([
                        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#0f0e0d")),
                        ("TEXTCOLOR",   (0,0), (-1,0), colors.HexColor("#f7f4ef")),
                        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
                        ("FONTSIZE",    (0,0), (-1,0), 7.5),
                        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#faf8f5")]),
                        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#e8e4dc")),
                        ("VALIGN",      (0,0), (-1,-1), "TOP"),
                        ("TOPPADDING",  (0,0), (-1,-1), 4),
                        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 8))

                # Citations
                urls = [e.get("url","") for e in ev if e.get("url") and e.get("url") != "tavily-synthesis"]
                if urls:
                    story.append(Paragraph("Citations", h2_style))
                    for i, url in enumerate(urls, 1):
                        story.append(Paragraph(
                            f"[{i}] {url}",
                            ParagraphStyle("cit", parent=body_style, fontSize=7.5, fontName="Courier", textColor=colors.HexColor("#1a5276")),
                        ))

                doc.build(story)
                return buf.getvalue()

            pdf_bytes = build_pdf(rpt)
            st.download_button(
                "⬇ Download Full Report (PDF)",
                data=pdf_bytes,
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )
        except Exception as pdf_err:
            st.warning(f"PDF generation failed ({pdf_err}). Falling back to JSON.")
            report_json = json.dumps(rpt, indent=2)
            st.download_button(
                "⬇ Download Full Report (JSON)",
                data=report_json,
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
            )


# ─── Right column: Audit trail ────────────────────────────────────────────────
with right:
    phase = st.session_state.phase
    status_map = {
        "idle":          ("idle",    "Idle"),
        "planning":      ("running", "Planning..."),
        "planned":       ("idle",    "Awaiting path selection"),
        "path_selected": ("idle",    "Ready to run"),
        "running":       ("running", "Running pipeline..."),
        "complete":      ("done",    "Complete"),
    }
    sc, sl = status_map.get(phase, ("idle", "Idle"))
    token_count = st.session_state.get("token_count", 0)
    model_cost = TOKEN_COST.get(model, None)
    cost_text = "n/a"
    if model_cost is not None:
        cost_text = f"${token_count * model_cost / 1000:.4f}"
    token_text = f"Tokens: {token_count:,}"
    status_extra = f"<span style=\"margin-left:1rem;font-size:0.85rem;color:#666;\">{token_text} · Cost: {cost_text}</span>"

    st.markdown(
        f'<div class="section-label">System Status &nbsp; <span class="status-pill status-{sc}">{sl}</span>{status_extra}</div>',
        unsafe_allow_html=True,
    )

    # Agent pipeline diagram
    if st.session_state.get("live_pipeline_html") and phase in ("running", "complete"):
        # Use live updating pipeline from session state
        st.markdown(st.session_state.live_pipeline_html, unsafe_allow_html=True)
    else:
        # Generate static pipeline diagram
        agents_list = [
            ("🗺", "Planner",   "Query decomposition"),
            ("🔍", "Retriever", "Multi-tool retrieval"),
            ("🧠", "Analyst",   "Contradiction detection"),
            ("💡", "Insight",   "Trend & hypothesis"),
            ("🎯", "Red Team",  "Hallucination check"),
            ("📋", "Reporter",  "Structured output"),
        ]
        agent_statuses = {}
        for step in st.session_state.audit_trail:
            agent_statuses[step["agent"]] = step["kind"]

        # If running, set the next agent to running
        if phase == "running":
            agent_order = ["PLANNER", "RETRIEVER", "ANALYST", "INSIGHT", "RED TEAM", "REPORTER"]
            for ag in agent_order:
                if ag not in agent_statuses or agent_statuses[ag] not in ["success", "error"]:
                    agent_statuses[ag] = "running"
                    break

        pipeline_html = '<div style="margin-bottom:1rem">'
        for icon, name, desc in agents_list:
            key = name.upper().replace(" ", " ")
            status = agent_statuses.get(key, "pending")
            if status == "success":
                dot = '●'; dot_color = "#3d6b52"; name_style = ""
            elif status == "error":
                dot = '●'; dot_color = "#b84c2e"; name_style = ""
            elif status == "running":
                dot = '●'; dot_color = "#d4871a"; name_style = 'style="color:#d4871a;font-weight:bold;"'  # Highlight name for running
            else:
                dot = '○'; dot_color = "#bbb"; name_style = ""
            pipeline_html += f'<div style="display:flex;align-items:center;gap:0.5rem;padding:0.35rem 0;font-size:0.82rem">'
            pipeline_html += f'<span style="color:{dot_color};font-size:0.9rem">{dot}</span>'
            pipeline_html += f'<span>{icon} <strong {name_style}>{name}</strong> <span style="color:#999;font-size:0.75rem">— {desc}</span></span></div>'
            if name != "Reporter":
                pipeline_html += '<div style="margin-left:0.55rem;width:1px;height:8px;background:#ddd"></div>'
        pipeline_html += '</div>'
        st.markdown(pipeline_html, unsafe_allow_html=True)

    # Audit trail
    trail = st.session_state.audit_trail
    if trail:
        st.markdown('<div class="section-label" style="margin-top:1rem">Audit Trail</div>', unsafe_allow_html=True)
        trail_html = ""
        for step in trail:
            k = step.get("kind", "info")
            css_class = f"audit-step step-{k}" if k in ("error", "success") else "audit-step"
            trail_html += (
                f'<div class="{css_class}">'
                f'<span class="step-time">{step["ts"]}</span>'
                f'<span class="step-agent">{step["agent"]}</span>'
                f'<span>{step["message"]}</span>'
                f'</div>'
            )
        st.markdown(trail_html, unsafe_allow_html=True)

    # Red team summary (right panel)
    if phase == "complete" and st.session_state.final_state:
        rpt = st.session_state.final_state.get("final_report", {})
        rt  = rpt.get("red_team_summary", {})
        if rt.get("validator_notes"):
            st.markdown('<div class="section-label" style="margin-top:1rem">Validator Notes</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:0.82rem;background:var(--validator-bg);color:var(--ink);padding:0.75rem;border:1px solid var(--validator-border)">{rt["validator_notes"]}</div>',
                unsafe_allow_html=True,
            )
        gaps = rt.get("logical_gaps", [])
        if gaps:
            st.markdown('<div class="section-label" style="margin-top:0.8rem">Logical Gaps</div>', unsafe_allow_html=True)
            for g in gaps:
                st.markdown(f'<div style="font-size:0.8rem;padding:0.25rem 0.5rem;border-left:2px solid #d4871a;margin-bottom:0.25rem">{g}</div>', unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #e0dcd5;text-align:center;font-family:monospace;font-size:0.68rem;color:#aaa;letter-spacing:0.1em;text-transform:uppercase">
  Research Intelligence System · LangGraph + OpenRouter · Multi-Agent Orchestration
</div>
""", unsafe_allow_html=True)
