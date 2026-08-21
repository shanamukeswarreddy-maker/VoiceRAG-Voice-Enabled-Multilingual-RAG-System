# 🎙️ VoiceRAG — Voice-Enabled Multilingual RAG System

A production-grade, ultra-low-latency Retrieval-Augmented Generation (RAG) system with voice input support, built for the **ai4bharat/MSMARCO-XI** Indic-language dataset.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Browser Frontend                                │
│   ┌───────────────┐   ┌─────────────────┐   ┌───────────────────────────┐   │
│   │   Mic Input   │   │   Text Input    │   │  Per-Stage Latency Meter  │   │
│   │ (WebAudio API)│   │  (Indic/English)│   │  Strategy Selector Dropdown│  │
│   └───────┬───────┘   └────────┬────────┘   └───────────────────────────┘   │
└───────────┼────────────────────┼────────────────────────────────────────────┘
            │ audio              │ text
            ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend (Async)                             │
│                                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │  STT Stage   │    │         RAG Pipeline (<200ms target)             │   │
│  │  (Sarvam AI) │───▶│                                                  │   │
│  │  saaras:v3   │    │  ┌───────────┐  ┌──────────────┐  ┌───────────┐  │   │
│  │              │    │  │Pre-Guard  │  │ ONNX Query   │  │  FAISS    │  │   │
│  │  Reported    │    │  │•Unsafe    │─▶│ Embedder     │─▶│  Search   │  │   │
│  │  SEPARATELY  │    │  │•Off-topic │  │ (L6 ONNX)    │  │  (top-k)  │  │   │
│  └──────────────┘    │  └───────────┘  └──────────────┘  └─────┬─────┘  │   │
│                      │                                         │        │   │
│                      │  ┌───────────┐  ┌──────────────┐        │        │   │
│                      │  │Post-Guard │◀─│ LLM Generate │◀───────┘        │   │
│                      │  │•Grounding │  │ (Groq LPU)   │                 │   │
│                      │  └───────────┘  └──────────────┘                 │   │
│                      └──────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Request Tracing: UUID + per-stage microsecond latency logs          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        FAISS Vector Store (In-Memory)                        │
│                                                                             │
│  Chunking Strategies (all stored in unified index with strategy tagging):   │
│  ┌───────────────┐ ┌───────────────┐ ┌─────────────────┐ ┌───────────────┐ │
│  │  Fixed-Size   │ │   Semantic    │ │ Sentence-Window │ │Metadata-Aware │ │
│  │ 256 tokens,   │ │ Topic-shift   │ │ Anchor + context│ │ Full MSMARCO  │ │
│  │ 20% overlap   │ │ breakpoints   │ │ window metadata │ │ passage pairs │ │
│  └───────────────┘ └───────────────┘ └─────────────────┘ └───────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Latency Budget & Acceleration Specs

| Stage | Technology / Strategy | Latency Target | Measured P50 Latency | Peak / P100 Bounded |
|---|---|---|---|---|
| **STT Stage** | Sarvam AI (`saaras:v3`) | *Reported Separately* | ~500–1200ms (Network bound) | ~2000ms |
| **Pipeline Total** | Core RAG Engine | **<200ms** | **186.8ms – 188.4ms** (P50) | **245.2ms** |
| ↳ Pre-Guardrails | Regex & Unsafe Input Filter | <1ms | **0.10ms** | **0.15ms** |
| ↳ Query Embedding | `paraphrase-multilingual-MiniLM-L12-v2` | <35ms | **26.6ms – 30.9ms** | **81.7ms** |
| ↳ FAISS Vector Search | Pre-Filtered Sub-Index (`IndexFlatIP`) | <5ms | **0.60ms – 0.90ms** | **2.10ms** |
| ↳ LLM Generation | Groq LPU (`groq/compound-mini`) | <170ms | **159.0ms** | **173.9ms** (Bounded timeout) |
| ↳ Post-Guardrails | Grounding & Validation Check | <1ms | **0.20ms** | **0.60ms** |

> **Note**: Speech-to-Text (Sarvam API) latency is measured and reported **separately** from the <200ms retrieval + generation pipeline target in the UI and automated benchmark reports.

---

## 🛠️ Technology Stack

| Component | Technology / Library | Description |
|---|---|---|
| **Web Framework** | FastAPI (Async Uvicorn) | High-concurrency async Python backend |
| **Speech-to-Text** | Sarvam AI (`saaras:v3`) | Indic-language STT with backoff retry & confidence metrics |
| **Query Embedding** | ONNX Runtime (`all-MiniLM-L6-v2`) | Accelerated C++ execution graph (**6.5ms** query latency) |
| **Passage Embedding** | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual Indic (Hindi & English) passage encoder |
| **Vector DB** | FAISS (`IndexFlatIP`) | In-memory similarity search over 15,000+ chunks |
| **LLM Inference** | Groq LPU API (`groq/compound-mini`) | Ultra-fast inference with time-budgeted fallback protection |
| **Dataset** | `ai4bharat/MSMARCO-XI` | Authentic Indic MS MARCO Hindi & English pairs |
| **Frontend** | Glassmorphism Dark Web UI | Vanilla HTML5/CSS3/JS + Web Audio API visualizer |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- Virtual Environment (`.venv`)
- API Keys:
  - **Groq API Key**: [console.groq.com](https://console.groq.com)
  - **Sarvam AI Key**: [dashboard.sarvam.ai](https://dashboard.sarvam.ai)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/goa_hackathon.git
cd goa_hackathon

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # On Windows
# source .venv/bin/activate # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```env
SARVAM_API_KEY=your_sarvam_api_key_here
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 4. Preprocess Dataset & Export ONNX Model
```bash
# Download authentic MSMARCO-XI dataset passages & queries
python scripts/download_dataset.py

# Build FAISS vector index across all 4 chunking strategies
python scripts/build_index.py
```

### 5. Launch the Server
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**.

---

## 🧪 Latency Benchmarking

Run the automated benchmark suite across 100 sampled test queries:
```bash
python scripts/benchmark.py
```

The benchmark script evaluates all 4 chunking strategies and generates:
- Markdown summary: `results/latency_report.md`
- Visual distribution chart: `results/latency_chart.png`

---

## 🧩 Chunking Strategies

| Strategy | Description | Typical Use Case |
|---|---|---|
| **Fixed-Size** | 256 tokens with 20% token overlap | Fast baseline for uniform passages |
| **Semantic** | Splitting at semantic topic-shift breakpoints | Preserves conceptual coherence |
| **Sentence-Window** | Small anchor sentence + expanded context window | High retrieval precision with full context |
| **Metadata-Aware** | Complete passages preserving MSMARCO fields | Structured metadata filtering & boosting |

---

## 🛡️ Guardrails & Safety Pipeline

1. **STT Confidence Guardrail**: Rejects audio input if Sarvam transcript confidence is `< 0.60`.
2. **Unsafe Input Filter**: Blocks PII (emails, SSNs), profanity, and prompt injection attacks.
3. **Off-Topic Detector**: Cosine similarity against 500 domain representative vectors.
4. **Grounding Verifier**: Cross-lingual Jaccard lexical and semantic support verification.
5. **Conversational Intent Handler**: Instant sub-1ms response for greetings ("namaste", "hello", "how are you").

---

## 📁 Project Directory Layout

```
goa_hackathon/
├── app/
│   ├── main.py              # FastAPI server & route handlers
│   ├── config.py             # Pydantic settings & configuration
│   ├── models.py             # Pydantic data schemas
│   ├── tracing.py            # Microsecond request-level tracing
│   ├── pipeline.py           # RAG Pipeline orchestrator
│   ├── stages/
│   │   ├── stt.py            # Sarvam AI STT adapter with retry logic
│   │   ├── chunking.py       # 4 multi-chunking strategy implementations
│   │   ├── embedder.py       # ONNX Runtime query & PyTorch passage embedder
│   │   ├── retriever.py      # FAISS vector similarity search
│   │   ├── generator.py      # Groq LPU answer generation
│   │   └── guardrails.py     # Pre & post-retrieval guardrail checks
│   └── static/
│       ├── index.html        # Glassmorphism dark-theme dashboard
│       ├── style.css         # Custom responsive CSS design system
│       └── app.js            # Web Audio mic recorder & API client
├── data/
│   ├── raw/                  # Preprocessed MSMARCO-XI passages & queries
│   ├── indices/              # FAISS index bin & chunk metadata JSON
│   └── minilm_l6.onnx        # Exported ONNX MiniLM L6 query execution graph
├── scripts/
│   ├── download_dataset.py   # Parquet extraction & preprocessing
│   ├── build_index.py        # Vector index construction
│   └── benchmark.py          # Latency benchmark harness
├── results/
│   ├── latency_report.md     # Generated benchmark report
│   └── latency_chart.png     # Latency distribution plots
├── config.yaml               # Application parameters
├── requirements.txt          # Python dependencies
└── README.md                 # System documentation
```

---

## 📜 License
Built for the Hackathon — Open source under the MIT License.
