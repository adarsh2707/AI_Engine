"""
Vantage RAG Pipeline
=====================
Enriches professional tasks with retrieved context before sending to models.

Stages:
  1. Embed the task (sentence-transformers, falls back to keyword matching)
  2. Retrieve similar past professional contexts from seed bank
  3. Rerank by relevance score
  4. Inject as grounding context into the model prompt

Install for full RAG:
  pip install sentence-transformers faiss-cpu

Without install: falls back to keyword-based retrieval (still useful).
"""

import re
import asyncio
import logging
from typing import Optional

logger = logging.getLogger("vantage.rag")

# ── Seed context bank ──────────────────────────────────────────────────────
# Real professional contexts used as retrieval corpus.
# In production: replace with a vector DB (Chroma, Pinecone, Supabase pgvector).

SEED_CONTEXTS = [
    # Data Science
    {"profession":"Data Scientist","topic":"class imbalance","context":"Class imbalance best practices: use stratified k-fold, evaluate on F1/AUC-PR not accuracy. XGBoost scale_pos_weight = negative/positive ratio. Threshold tuning on PR curve before SMOTE. SHAP values to verify feature importance before changing models. Leading churn indicators (usage drop 30d before) outperform lagging ones."},
    {"profession":"Data Scientist","topic":"feature selection","context":"High-dimensional feature selection: LASSO for linear, tree-based importance for nonlinear. RFECV for recursive elimination. VIF > 10 indicates multicollinearity. PCA for dense correlated features. Beware data leakage — never include post-event features in churn prediction."},
    {"profession":"Data Scientist","topic":"model deployment","context":"MLOps: version with MLflow, containerise with Docker, serve with FastAPI. Monitor data drift (KS test), concept drift (performance degradation), prediction distribution shift. Alert on PSI > 0.2. Shadow deployment before full rollout. Retrain triggers: performance threshold or scheduled (weekly/monthly)."},
    # Software Engineering
    {"profession":"Software Engineer","topic":"caching","context":"Cache stampede prevention: TTL jitter (±10-20%), probabilistic early expiration (PER), mutex on cache miss. Redlock for distributed locking (3+ Redis nodes). Write-through for consistency, write-behind for throughput. Cache-aside safest for read-heavy. Invalidate on writes, not on reads."},
    {"profession":"Software Engineer","topic":"system design","context":"High throughput API: horizontal scaling behind load balancer, stateless services, connection pooling (PgBouncer). Rate limiting with token bucket (Redis). Async processing for non-critical paths. Read replicas for read-heavy. CDN for static. Circuit breaker for external dependencies."},
    {"profession":"Software Engineer","topic":"database","context":"Index on WHERE, JOIN, ORDER BY columns. Composite indexes: selectivity first. Partial indexes for filtered queries. EXPLAIN ANALYZE to catch sequential scans. N+1: use eager loading. Keyset pagination over OFFSET for large datasets. Partition tables by date for time-series data."},
    # Contract Attorney
    {"profession":"Contract Attorney","topic":"indemnification","context":"Indemnification: mutual for commercial contracts, cap at contract value, carve-outs for gross negligence/wilful misconduct, contributory negligence reduction. Delaware courts read 'arising out of or relating to' broadly — avoid unless intended. Survival beyond termination must be explicit. IP indemnification should have separate, higher cap."},
    {"profession":"Contract Attorney","topic":"liability","context":"LOL structure: mutual cap at 12-month fees standard for SaaS. Separate higher cap for IP and confidentiality. Uncapped carve-outs: fraud, death/personal injury, wilful misconduct. Consequential damages waiver: exclude both parties' lost profits, revenue, data. Check jurisdiction — some states limit LOL enforceability."},
    {"profession":"Contract Attorney","topic":"force majeure","context":"Post-COVID force majeure: enumerate pandemic, government mandates, supply chain disruption explicitly. Include: notice obligation (5-10 business days), mitigation duty, termination right if delay exceeds 90 days. Distinguish suspension vs termination. Payment obligations during FM period must be addressed separately."},
    # Financial Analyst
    {"profession":"Financial Analyst","topic":"valuation","context":"DCF: WACC from current market rates, TGR ≤ GDP growth (1-3%). Sensitivity: ±150bps WACC, ±1.5% TGR. EV/EBITDA cross-check: exclude negative EBITDA comps. Two-stage DCF for high-growth. Contamination risk: pro-forma adjustments for recent M&A. NTM preferred over LTM for fast-growing companies."},
    {"profession":"Financial Analyst","topic":"comps","context":"Comps selection: same industry, similar size (±50% revenue), similar growth profile. Metrics: EV/Revenue (early stage), EV/EBITDA (mature), P/E (profitable). Normalize for non-recurring items, lease accounting. Median over mean to reduce outlier influence. Flag comps with recent M&A activity."},
    # Product Manager
    {"profession":"Product Manager","topic":"prioritization","context":"Prioritization frameworks: RICE (Reach × Impact × Confidence / Effort), ICE, MoSCoW. Separate discovery from delivery. User research before roadmapping. OKRs: objective is qualitative, key results are measurable and time-bound. Avoid output metrics (features shipped) in favor of outcome metrics (retention, activation)."},
    {"profession":"Product Manager","topic":"metrics","context":"North Star Metric: single metric capturing product value delivery. Input metrics: actions users take. Output metrics: business results. Avoid vanity metrics (pageviews, downloads). Cohort analysis for retention. Funnel analysis for conversion. A/B test minimum detectable effect before running experiments."},
    # HR Manager
    {"profession":"HR Manager","topic":"performance","context":"Performance management: separate development conversations from compensation decisions. 360 feedback: structured, anonymous, actionable. PIP (Performance Improvement Plan): specific, measurable, time-bound goals, documented support provided. Document everything. Involve legal before termination. Constructive dismissal risk from unilateral role changes."},
    {"profession":"HR Manager","topic":"compensation","context":"Compensation bands: market data (Radford, Levels.fyi, Glassdoor), job architecture, internal equity. Pay transparency laws vary by state/country. Equity refresh grants for retention. Variable pay: clearly defined metrics, achievable targets (80-120% of quota). Total compensation statement for retention conversations."},
    # ML Engineer
    {"profession":"ML Engineer","topic":"inference","context":"Inference optimization: quantization (INT8, FP16), pruning, knowledge distillation. Batch inference for throughput, streaming for latency. Model serving: TorchServe, Triton, vLLM for LLMs. SLA: p50/p99 latency, not average. GPU utilization target: 70-80%. Model versioning with A/B traffic splitting."},
    {"profession":"ML Engineer","topic":"training","context":"Distributed training: data parallelism (DDP), model parallelism for large models. Mixed precision (AMP) for 2x speedup. Gradient checkpointing for memory. Learning rate warmup + cosine decay. Early stopping on validation loss. Checkpoint every N steps. Experiment tracking: MLflow, W&B. Reproducibility: seed everything."},
]

# ── Keyword retrieval fallback ─────────────────────────────────────────────

def keyword_score(task: str, ctx: dict) -> float:
    """Simple TF-based relevance when embeddings unavailable."""
    task_lower = task.lower()
    topic_words = ctx["topic"].lower().split()
    context_words = set(ctx["context"].lower().split())
    task_words = set(task_lower.split())

    topic_match = sum(1 for w in topic_words if w in task_lower) / max(len(topic_words), 1)
    overlap = len(task_words & context_words) / max(len(task_words), 1)
    return topic_match * 0.7 + overlap * 0.3


# ── RAG Pipeline ───────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self):
        self._ready = False
        self._attempted = False
        self._embedder = None
        self._index = None
        self._np = None

    def _try_init(self):
        if self._attempted:
            return
        self._attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np

            self._np = np
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

            texts = [f"{c['topic']} {c['context']}" for c in SEED_CONTEXTS]
            embeddings = self._embedder.encode(texts, convert_to_numpy=True)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(embeddings.astype("float32"))

            self._ready = True
            logger.info("✓ RAG pipeline ready — %d contexts indexed", len(SEED_CONTEXTS))
        except ImportError:
            logger.info("ℹ RAG using keyword fallback — install sentence-transformers faiss-cpu for full RAG")
        except Exception as e:
            logger.warning("RAG init failed: %s", e)

    async def enrich(self, task: str, profession: str, top_k: int = 2) -> tuple[str, list]:
        """
        Returns (enriched_prompt, retrieved_contexts).
        Falls back gracefully if RAG unavailable.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._enrich_sync(task, profession, top_k))

    def _enrich_sync(self, task: str, profession: str, top_k: int) -> tuple[str, list]:
        self._try_init()

        # Filter by profession first
        prof_contexts = [c for c in SEED_CONTEXTS if c["profession"] == profession]
        all_contexts = prof_contexts + [c for c in SEED_CONTEXTS if c["profession"] != profession]

        retrieved = []

        if self._ready:
            try:
                import numpy as np
                q = self._embedder.encode([task], convert_to_numpy=True)
                q = q / np.linalg.norm(q, axis=1, keepdims=True)
                scores, indices = self._index.search(q.astype("float32"), min(top_k + 3, len(SEED_CONTEXTS)))
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or score < 0.25: continue
                    ctx = SEED_CONTEXTS[idx]
                    if ctx["profession"] == profession or score > 0.55:
                        retrieved.append((float(score), ctx))
                    if len(retrieved) >= top_k: break
            except Exception as e:
                logger.debug("FAISS search failed: %s", e)

        # Keyword fallback
        if not retrieved:
            scored = [(keyword_score(task, c), c) for c in all_contexts]
            scored.sort(key=lambda x: x[0], reverse=True)
            retrieved = [(s, c) for s, c in scored[:top_k] if s > 0.05]

        if not retrieved:
            return task, []

        context_block = "\n\n".join([
            f"[Professional context — {c['profession']} / {c['topic']}]\n{c['context']}"
            for _, c in retrieved
        ])

        enriched = f"""{task}

---
Retrieved professional context (use where relevant, ignore if not applicable):
{context_block}
---"""

        return enriched, [c for _, c in retrieved]

    def status(self) -> dict:
        self._try_init()
        return {
            "ready": self._ready,
            "mode": "faiss" if self._ready else "keyword",
            "contexts_indexed": len(SEED_CONTEXTS),
        }


rag = RAGPipeline()
