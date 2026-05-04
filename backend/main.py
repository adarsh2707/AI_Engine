"""
Vantage of AI — Backend v6
===========================
Incorporates his architecture:
  ✓ Async-native model calls (AsyncAnthropic, AsyncOpenAI)
  ✓ Presidio NER-based PII masking (fallback to regex)
  ✓ Vertical-specific prompt templates (his templates.py approach)
  ✓ Bias checker (pure Python, no deps)
  ✓ Latency tracking per model call
  ✓ Unified model runner with semaphore concurrency control
  ✓ All v5 features: 24 professions, PCI, SQLite, leaderboard, live counter

Install:
  pip install fastapi uvicorn google-generativeai openai anthropic
      python-dotenv langchain-google-genai
  pip install presidio-analyzer presidio-anonymizer  (optional, stronger PII)
  python -m spacy download en_core_web_lg            (optional, for Presidio)

Run:
  uvicorn main:app --reload --port 8000
"""

import os, json, asyncio, uuid, random, sqlite3, re, time, logging, math
from datetime import datetime
from typing import Optional
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("vantage")

app = FastAPI(title="Vantage of AI", version="6.0.0")
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── SQLite ─────────────────────────────────────────────────────────────────

DB_PATH = "vantage.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            profession TEXT,
            task TEXT,
            task_masked TEXT,
            task_type TEXT,
            responses TEXT,
            vote INTEGER,
            winner_model TEXT,
            winner_latency_ms REAL,
            dimension_scores TEXT,
            pci_score REAL,
            bias_flags TEXT,
            pii_count INTEGER DEFAULT 0,
            created_at TEXT,
            voted_at TEXT
        );
        CREATE TABLE IF NOT EXISTS dimension_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            profession TEXT,
            winner_model TEXT,
            dimension TEXT,
            score INTEGER,
            created_at TEXT
        );
        """)

init_db()

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

# ── Model registry (his approach) ─────────────────────────────────────────

MODEL_REGISTRY = {
    "gemini-2.0-flash": {
        "display":  "Gemini 2.0 Flash",
        "provider": "Google",
        "env_key":  "GEMINI_API_KEY",
        "color":    "#4285F4",
        "context_window": 1_000_000,
    },
    "gpt-4o": {
        "display":  "GPT-4o",
        "provider": "OpenAI",
        "env_key":  "OPENAI_API_KEY",
        "color":    "#10A37F",
        "context_window": 128_000,
    },
    "claude-sonnet-4-20250514": {
        "display":  "Claude Sonnet",
        "provider": "Anthropic",
        "env_key":  "ANTHROPIC_API_KEY",
        "color":    "#D97757",
        "context_window": 200_000,
    },
    "mistral-large-latest": {
        "display":  "Mistral Large",
        "provider": "Mistral",
        "env_key":  "MISTRAL_API_KEY",
        "color":    "#FF7000",
        "context_window": 128_000,
    },
}

DEFAULT_MODELS = ["gemini-2.0-flash", "gpt-4o", "claude-sonnet-4-20250514"]

def available_models() -> list:
    return [{"id": k, **v, "live": bool(os.getenv(v["env_key"]))} for k, v in MODEL_REGISTRY.items()]

def is_available(model_id: str) -> bool:
    return bool(MODEL_REGISTRY.get(model_id) and os.getenv(MODEL_REGISTRY[model_id]["env_key"]))

# ── Vertical-specific prompt templates (from his templates.py) ─────────────

VERTICAL_TEMPLATES = {
    "Contract Attorney": """You are a legal AI assistant being evaluated by verified legal professionals.

Instructions:
- Base your analysis ONLY on the provided context and question.
- Distinguish clearly between what documents say and legal interpretation.
- Flag jurisdictional assumptions explicitly (e.g., "Under Delaware law...").
- Cite specific clauses, sections, or statutory references where applicable.
- Do not provide definitive legal advice — this is analysis for professional review.
- If evidence is insufficient, state this rather than speculating.""",

    "Radiologist": """You are a clinical AI assistant being evaluated by verified healthcare professionals.

Instructions:
- Base your response strictly on clinical evidence and provided context.
- Flag any statements you are uncertain about with [LOW CONFIDENCE].
- Use ICD-10/CPT codes where clinically appropriate.
- Never fabricate clinical facts or diagnostic conclusions.
- Treat all information as protected health information.""",

    "Nurse Practitioner": """You are a clinical AI assistant being evaluated by verified nurse practitioners.

Instructions:
- Ground all recommendations in evidence-based clinical practice.
- Flag drug interactions, contraindications, or safety concerns explicitly.
- Note when referral or specialist consultation is indicated.
- Never fabricate clinical data.""",

    "Financial Analyst": """You are a financial AI assistant being evaluated by verified finance professionals.

Instructions:
- Ground all analysis in the provided financial data and context.
- Cite specific figures, ratios, or data points.
- Flag any extrapolations clearly (e.g., "If we assume X...").
- Apply relevant frameworks (GAAP, IFRS, DCF) explicitly.
- Quantify uncertainty where possible.""",

    "Accountant / CPA": """You are an accounting AI assistant being evaluated by certified accounting professionals.

Instructions:
- Apply the correct accounting standards (GAAP/IFRS) explicitly.
- Reference specific line items, accounts, or transactions.
- Flag areas where professional judgment is required.
- Do not fabricate figures not present in the context.""",

    "Software Engineer": """You are a software engineering AI assistant being evaluated by verified engineers.

Instructions:
- Provide specific, actionable technical guidance.
- Explain reasoning behind recommendations.
- Flag trade-offs explicitly (performance vs maintainability, consistency vs availability).
- Reference design patterns, complexity implications, and failure modes.""",

    "ML Engineer": """You are an ML engineering AI assistant being evaluated by verified ML engineers.

Instructions:
- Ground recommendations in production ML best practices.
- Distinguish between research settings and production constraints.
- Flag latency, memory, and scalability implications.
- Reference specific frameworks, tools, and metrics where appropriate.""",

    "Data Scientist": """You are a data science AI assistant being evaluated by verified data scientists.

Instructions:
- Ground statistical recommendations in rigorous methodology.
- Flag assumptions about data distributions, independence, and stationarity.
- Distinguish between correlation and causation explicitly.
- Cite relevant statistical tests, their assumptions, and appropriate alternatives.""",

    "HR Manager": """You are an HR AI assistant being evaluated by HR professionals.

Instructions:
- Apply employment law principles relevant to the stated jurisdiction.
- Flag potential bias, equity, or legal compliance concerns proactively.
- Maintain strict confidentiality framing — treat all data as sensitive.
- Cite specific policy frameworks or legal standards where applicable.""",

    "Product Manager": """You are a product management AI assistant being evaluated by verified PMs.

Instructions:
- Ground recommendations in user research, data, and business context provided.
- Flag assumptions about user behaviour explicitly.
- Distinguish between qualitative insight and quantitative evidence.
- Be specific about metrics, success criteria, and measurability.""",
}

DEFAULT_TEMPLATE = """You are a domain expert AI assistant being evaluated by verified professionals.

Instructions:
- Be specific, actionable, and grounded in domain knowledge.
- Flag uncertainty rather than guessing.
- Cite reasoning behind recommendations.
- Keep response under 200 words and professionally precise."""

USER_TEMPLATE = "Professional task:\n{task}\n\nResponse:"

def get_system_prompt(profession: str, persona: str) -> str:
    """Return the best system prompt for this profession."""
    template = VERTICAL_TEMPLATES.get(profession, DEFAULT_TEMPLATE)
    return f"{persona}\n\n{template}"

# ── Bias checker (his bias_checker.py) ────────────────────────────────────

_MALE = re.compile(r"\b(he|him|his|himself)\b", re.IGNORECASE)
_FEMALE = re.compile(r"\b(she|her|hers|herself)\b", re.IGNORECASE)
_POSITIVE = re.compile(r"\b(excellent|strong|competent|qualified|capable|skilled|superior)\b", re.IGNORECASE)
_NEGATIVE = re.compile(r"\b(weak|incompetent|unqualified|inferior|risky|problematic|difficult)\b", re.IGNORECASE)
_DEMO_GROUPS = ["men","women","male","female","elderly","young","minority","white","black","asian","hispanic"]

def check_bias(text: str) -> dict:
    flags = []
    male_n = len(_MALE.findall(text))
    female_n = len(_FEMALE.findall(text))
    total = male_n + female_n
    if total > 4:
        ratio = male_n / (total + 1e-6)
        if ratio > 0.85:
            flags.append("Predominantly male pronouns")
        elif ratio < 0.15:
            flags.append("Predominantly female pronouns")
    text_l = text.lower()
    for group in _DEMO_GROUPS:
        if group in text_l:
            idx = text_l.find(group)
            window = text_l[max(0,idx-50):idx+len(group)+50]
            if len(_NEGATIVE.findall(window)) > 1:
                flags.append(f"Negative sentiment near '{group}'")
    return {"has_bias": len(flags) > 0, "flags": flags,
            "gender": {"male": male_n, "female": female_n}}

# ── PII masking (Presidio with regex fallback) ─────────────────────────────

_PII_PATTERNS = [
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL]'),
    (re.compile(r'\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'), '[PHONE]'),
    (re.compile(r'https?://\S+|www\.\S+'), '[URL]'),
    (re.compile(r'\b(Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?'), '[NAME]'),
    (re.compile(r'\bNPI[:\s#]*\d{10}\b', re.IGNORECASE), '[NPI]'),
    (re.compile(r'\bBar[:\s#]*\d{4,8}\b', re.IGNORECASE), '[BAR_NUMBER]'),
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN]'),
]

_presidio_analyzer = None
_presidio_anonymizer = None

def _load_presidio():
    global _presidio_analyzer, _presidio_anonymizer
    if _presidio_analyzer:
        return True
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        _presidio_analyzer = AnalyzerEngine()
        _presidio_anonymizer = AnonymizerEngine()
        logger.info("✓ Presidio PII masking loaded")
        return True
    except ImportError:
        return False

def mask_pii(text: str) -> tuple[str, int]:
    """Returns (masked_text, count_of_items_masked). Uses Presidio if available."""
    if _load_presidio():
        try:
            results = _presidio_analyzer.analyze(
                text=text, language="en",
                entities=["PERSON","EMAIL_ADDRESS","PHONE_NUMBER","CREDIT_CARD",
                          "MEDICAL_LICENSE","US_SSN","LOCATION","IP_ADDRESS","URL"])
            if not results:
                return text, 0
            from presidio_anonymizer import AnonymizerEngine
            anonymized = _presidio_anonymizer.anonymize(text=text, analyzer_results=results)
            return anonymized.text, len(results)
        except Exception as e:
            logger.debug("Presidio failed, using regex: %s", e)

    # Regex fallback
    masked, count = text, 0
    for pattern, replacement in _PII_PATTERNS:
        new = pattern.sub(replacement, masked)
        count += len(pattern.findall(masked))
        masked = new
    return masked, count

# ── Async model runner (his runner.py approach) ────────────────────────────

_semaphore = asyncio.Semaphore(10)  # max 10 concurrent LLM calls

async def call_model_async(model_id: str, system: str, user: str, max_tokens: int = 400) -> tuple[str, float]:
    """
    Async-native model call. Returns (response_text, latency_ms).
    Uses AsyncAnthropic / AsyncOpenAI for true async (not run_in_executor).
    """
    info = MODEL_REGISTRY.get(model_id, {})
    provider = info.get("provider", "")
    start = time.perf_counter()

    async with _semaphore:
        try:
            if provider == "Google":
                from google import genai as google_genai
                client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                loop = asyncio.get_event_loop()
                full = f"{system}\n\n{user}"
                resp = await loop.run_in_executor(None, lambda: client.models.generate_content(
                    model=model_id, contents=full))
                text = resp.text.strip()

            elif provider == "OpenAI":
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                resp = await client.chat.completions.create(
                    model=model_id, max_tokens=max_tokens,
                    messages=[{"role":"system","content":system},
                               {"role":"user","content":user}]
                )
                text = resp.choices[0].message.content.strip()

            elif provider == "Anthropic":
                import anthropic
                client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                msg = await client.messages.create(
                    model=model_id, max_tokens=max_tokens,
                    system=[{"type":"text","text":system,
                              "cache_control":{"type":"ephemeral"}}],  # prompt caching
                    messages=[{"role":"user","content":user}]
                )
                text = msg.content[0].text.strip()

            elif provider == "Mistral":
                from mistralai import Mistral
                client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(None, lambda: client.chat.complete(
                    model=model_id, max_tokens=max_tokens,
                    messages=[{"role":"system","content":system},
                               {"role":"user","content":user}]
                ))
                text = resp.choices[0].message.content.strip()

            else:
                text = f"[Unknown provider: {provider}]"

        except Exception as e:
            logger.error("Model %s error: %s", model_id, e)
            text = f"[{info.get('display', model_id)} temporarily unavailable]"

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info("Model %s responded in %.0fms", model_id, latency_ms)
    return text, latency_ms

# ── Professions ────────────────────────────────────────────────────────────

PROFESSIONS = {
    "Contract Attorney":      {"group":"Professional","dimensions":["Legal accuracy","Risk identification","Practical advice"],"task_types":["Contract review","Risk assessment","Negotiation strategy","Compliance","Dispute resolution"],"credential_label":"Bar number","prompt_persona":"You are a senior contract attorney with 12 years of corporate law experience."},
    "Radiologist":            {"group":"Professional","dimensions":["Clinical accuracy","Differential completeness","Safety awareness"],"task_types":["Image interpretation","Differential diagnosis","Report writing","Protocol selection","Findings correlation"],"credential_label":"NPI number","prompt_persona":"You are a board-certified radiologist with subspecialty training in diagnostic imaging."},
    "Financial Analyst":      {"group":"Professional","dimensions":["Analytical rigor","Model accuracy","Business insight"],"task_types":["Valuation","Financial modeling","Due diligence","Market analysis","Risk modeling"],"credential_label":"CFA / CRD number","prompt_persona":"You are a CFA-certified senior financial analyst at a top-tier investment bank."},
    "Accountant / CPA":       {"group":"Professional","dimensions":["Regulatory accuracy","Practical application","Risk flagging"],"task_types":["Tax planning","Financial reporting","Audit prep","Compliance","Advisory"],"credential_label":"CPA license number","prompt_persona":"You are a CPA with 10 years in corporate accounting and tax advisory."},
    "HR Manager":             {"group":"Professional","dimensions":["Policy correctness","Empathy & tone","Actionability"],"task_types":["Policy writing","Performance management","Conflict resolution","Hiring & onboarding","Compensation"],"credential_label":"SHRM credential","prompt_persona":"You are an experienced HR manager who built people ops at two high-growth startups."},
    "Journalist":             {"group":"Professional","dimensions":["Factual accuracy","Source quality","Narrative clarity"],"task_types":["Story angle","Research","Headline writing","Interview prep","Fact-checking"],"credential_label":"Press credential","prompt_persona":"You are an investigative journalist with 10 years at a national publication."},
    "Nurse Practitioner":     {"group":"Professional","dimensions":["Clinical accuracy","Patient safety","Evidence-based reasoning"],"task_types":["Patient assessment","Treatment planning","Medication management","Documentation","Patient education"],"credential_label":"NPI number","prompt_persona":"You are a board-certified nurse practitioner with 8 years in primary care."},
    "Insurance Underwriter":  {"group":"Professional","dimensions":["Risk accuracy","Policy correctness","Actuarial reasoning"],"task_types":["Risk assessment","Policy pricing","Claims review","Compliance","Portfolio analysis"],"credential_label":"CPCU credential","prompt_persona":"You are a senior insurance underwriter specialising in commercial property and casualty."},
    "Software Engineer":      {"group":"Technical","dimensions":["Technical accuracy","Code clarity","Scalability thinking"],"task_types":["System design","Code review","Debugging","Architecture","Performance"],"credential_label":"GitHub profile","prompt_persona":"You are a senior software engineer with deep expertise in distributed systems and production engineering."},
    "ML Engineer":            {"group":"Technical","dimensions":["Model correctness","Production readiness","Efficiency"],"task_types":["Model optimization","Pipeline design","Serving & inference","Monitoring","Training infrastructure"],"credential_label":"GitHub profile","prompt_persona":"You are a senior ML engineer who builds and ships production ML systems at scale."},
    "Data Scientist":         {"group":"Technical","dimensions":["Statistical correctness","Code quality","Assumption clarity"],"task_types":["Model selection","EDA & visualization","Feature engineering","Evaluation & metrics","Deployment & MLOps"],"credential_label":"GitHub / Kaggle profile","prompt_persona":"You are a senior data scientist with 8 years in ML and statistical modeling."},
    "DevOps / Platform Eng":  {"group":"Technical","dimensions":["Infrastructure correctness","Security awareness","Reliability thinking"],"task_types":["CI/CD pipelines","Cloud architecture","Incident response","Cost optimisation","Security hardening"],"credential_label":"GitHub profile","prompt_persona":"You are a senior platform engineer with expertise in Kubernetes, Terraform, and cloud-native infrastructure."},
    "Cybersecurity Analyst":  {"group":"Technical","dimensions":["Threat accuracy","Remediation quality","Risk communication"],"task_types":["Threat modeling","Incident response","Vulnerability assessment","Policy writing","Penetration testing"],"credential_label":"CISSP / CEH credential","prompt_persona":"You are a senior cybersecurity analyst with expertise in threat detection and incident response."},
    "Data Engineer":          {"group":"Technical","dimensions":["Pipeline correctness","Data quality","Scalability"],"task_types":["Pipeline design","Data modeling","ETL/ELT","Data quality","Query optimisation"],"credential_label":"GitHub profile","prompt_persona":"You are a senior data engineer who builds and maintains large-scale data pipelines."},
    "UX Designer":            {"group":"Creative","dimensions":["User empathy","Design rationale","Feasibility"],"task_types":["User research","Wireframing","Usability review","Design systems","Accessibility"],"credential_label":"Portfolio URL","prompt_persona":"You are a senior UX designer with a portfolio spanning fintech, healthtech, and consumer apps."},
    "Marketing Manager":      {"group":"Creative","dimensions":["Strategic clarity","Audience targeting","Measurable outcomes"],"task_types":["Campaign strategy","Copy & messaging","Channel selection","Analytics & attribution","Brand positioning"],"credential_label":"LinkedIn profile","prompt_persona":"You are a senior marketing manager who has led growth at multiple B2B SaaS companies."},
    "Copywriter":             {"group":"Creative","dimensions":["Persuasiveness","Brand alignment","Originality"],"task_types":["Ad copy","Email campaigns","Landing pages","Social content","Brand voice"],"credential_label":"Portfolio URL","prompt_persona":"You are a senior copywriter with 8 years across DTC, SaaS, and agency work."},
    "Technical Writer":       {"group":"Creative","dimensions":["Clarity","Accuracy","User-appropriateness"],"task_types":["API documentation","User guides","Release notes","SOPs","Knowledge base"],"credential_label":"Portfolio URL","prompt_persona":"You are a senior technical writer with expertise in developer documentation and API references."},
    "Product Manager":        {"group":"Operational","dimensions":["User focus","Prioritization logic","Measurability"],"task_types":["PRD writing","Prioritization","Metrics & OKRs","User research","Roadmapping"],"credential_label":"LinkedIn profile","prompt_persona":"You are a senior product manager with a track record of 0→1 products at Series B companies."},
    "Operations Manager":     {"group":"Operational","dimensions":["Process clarity","Efficiency focus","Risk awareness"],"task_types":["Process design","KPI tracking","Vendor management","Cost reduction","Team coordination"],"credential_label":"LinkedIn profile","prompt_persona":"You are a senior operations manager who has scaled processes from startup to growth stage."},
    "Supply Chain Analyst":   {"group":"Operational","dimensions":["Quantitative accuracy","Risk identification","Practical recommendations"],"task_types":["Demand forecasting","Inventory optimisation","Supplier analysis","Logistics","Risk mitigation"],"credential_label":"APICS credential","prompt_persona":"You are a senior supply chain analyst with expertise in forecasting, inventory, and logistics."},
    "Real Estate Analyst":    {"group":"Operational","dimensions":["Market accuracy","Financial rigor","Risk assessment"],"task_types":["Property valuation","Market analysis","Investment modeling","Due diligence","Portfolio analysis"],"credential_label":"License number","prompt_persona":"You are a senior real estate analyst specialising in commercial property valuation."},
    "Customer Success Mgr":   {"group":"Operational","dimensions":["Empathy & tone","Problem-solving","Retention focus"],"task_types":["Churn prevention","Onboarding","QBR preparation","Escalation handling","Expansion planning"],"credential_label":"LinkedIn profile","prompt_persona":"You are a senior customer success manager managing enterprise accounts at a SaaS company."},
    "Executive Assistant":    {"group":"Operational","dimensions":["Clarity","Professionalism","Attention to detail"],"task_types":["Email drafting","Meeting prep","Travel coordination","Document management","Stakeholder communication"],"credential_label":"LinkedIn profile","prompt_persona":"You are a senior executive assistant supporting C-suite leaders at a Fortune 500 company."},
}

# ── PCI scoring ────────────────────────────────────────────────────────────

def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total == 0: return 0.0
    p = wins / total
    denom = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    margin = z * ((p * (1-p) / total + z**2 / (4 * total**2)) ** 0.5)
    return max(0.0, (centre - margin) / denom)

REFERENCE_SCORES = {
    "Gemini 2.0 Flash": {"abstract":0.72,"knowledge":0.81,"math":0.85,"code":0.80,"swe":0.74,"truth":0.70,"science":0.78,"terminal":0.65,"frontier":0.73,"web":0.68,"tool":0.75},
    "GPT-4o":           {"abstract":0.79,"knowledge":0.88,"math":0.92,"code":0.88,"swe":0.85,"truth":0.75,"science":0.84,"terminal":0.72,"frontier":0.82,"web":0.76,"tool":0.83},
    "Claude Sonnet":    {"abstract":0.75,"knowledge":0.85,"math":0.87,"code":0.84,"swe":0.80,"truth":0.82,"science":0.81,"terminal":0.70,"frontier":0.78,"web":0.72,"tool":0.79},
    "Mistral Large":    {"abstract":0.70,"knowledge":0.80,"math":0.82,"code":0.79,"swe":0.73,"truth":0.71,"science":0.76,"terminal":0.63,"frontier":0.70,"web":0.64,"tool":0.72},
}

VERTICAL_BIAS = {
    "Contract Attorney":  {"truth":1.7,"knowledge":1.3},
    "Radiologist":        {"truth":1.6,"science":1.4},
    "Software Engineer":  {"swe":1.8,"code":1.5},
    "ML Engineer":        {"code":1.6,"swe":1.4},
    "Data Scientist":     {"math":1.4,"code":1.3},
    "Financial Analyst":  {"math":1.5,"frontier":1.2},
    "Accountant / CPA":   {"math":1.7,"truth":1.4},
    "HR Manager":         {"truth":1.5,"knowledge":1.2},
}

def compute_pci(model_display: str, profession: str, wins: int, total: int) -> float:
    ref = REFERENCE_SCORES.get(model_display, {})
    weights = {"abstract":0.05,"knowledge":0.05,"math":0.05,"code":0.04,"swe":0.06,"truth":0.05,"science":0.04,"terminal":0.05,"frontier":0.04,"web":0.05,"tool":0.05,"prof":0.20,"real":0.15,"rigor":0.10}
    bias = VERTICAL_BIAS.get(profession, {})
    for k, mult in bias.items():
        if k in weights: weights[k] *= mult
    total_w = sum(weights.values())
    weights = {k: v/total_w for k, v in weights.items()}
    public = sum(ref.get(k,0.70) * weights[k] for k in ref)
    wr = wilson_lower_bound(wins, total)
    prof = wr * weights["prof"]
    real = wr * weights["real"]
    confidence = min(1.0, total / 50)
    rigor = confidence * wr * weights["rigor"]
    return round(min(1.0, public + prof + real + rigor), 4)

# ── Mock responses (for mock mode) ────────────────────────────────────────

MOCK_RESPONSES = {
    "Data Scientist": {
        "gemini-2.0-flash": "Switch to XGBoost with scale_pos_weight=11.5 — fastest path to better recall. Lower classification threshold to 0.2–0.3 using precision-recall curve, not ROC. Check SHAP values first — if churn features have low importance, the bottleneck is feature engineering, not the model. Only add SMOTE if recall stays below 65% after both changes.",
        "gpt-4o": "Three steps:\n\n1. Threshold tuning — move from 0.5 to 0.25 using the PR curve. Expect recall to jump to 65–70%.\n2. XGBoost with class_weight='balanced', scale_pos_weight ~11.5.\n3. SMOTE only as last resort — synthetic samples don't reflect real churn patterns.\n\nUse stratified k-fold with F1 throughout. Accuracy at 8% imbalance is meaningless.",
        "claude-sonnet-4-20250514": "Before changing models, run SHAP on your logistic regression. If churn signals (session drop, plan downgrade, support tickets) have low importance, no model change will help — fix features first.\n\nIf features look healthy: threshold adjustment to 0.25 is your fastest win, then XGBoost. Avoid SMOTE unless below 1% churn rate.\n\nSwitch your evaluation metric to F1 or AUC-PR immediately. Accuracy at 8% imbalance is not a signal.",
    },
    "Software Engineer": {
        "gemini-2.0-flash": "Write-through with Redlock (3 Redis nodes) and 300s TTL ±30s jitter. Circuit breaker for Redis failures. ~2ms latency overhead, stampede risk eliminated.",
        "gpt-4o": "Three options by complexity:\n1. Write-through + jitter: 300s ±30s. Eliminates thundering herd.\n2. Cache-aside + optimistic locking: compare-and-set with version tag.\n3. Redlock: correct but operationally heavier.\n\nAlso: 60s TTL is too aggressive — ~1.6M DB hits/day. Bump to 600s, invalidate on writes.",
        "claude-sonnet-4-20250514": "Stampede and race condition are separate problems.\n\nFor stampede: TTL jitter (±30s on 300s) covers 90% of cases. Add probabilistic early expiration for the rest.\n\nFor race condition: cache-aside with optimistic locking (compare-and-set + version tag).\n\nRedlock is the nuclear option — correct but adds operational complexity. Reach for it only if the above don't work.",
    },
    "Contract Attorney": {
        "gemini-2.0-flash": "§8.2 is overbroad — 'arising out of or relating to' covers indemnitee's own negligence under Delaware law. Narrow to: material breach, gross negligence, wilful misconduct. Add contributory negligence carve-out.",
        "gpt-4o": "Three issues:\n1. §8.2 indemnification — too broad. Narrow trigger to material breach, gross negligence, wilful misconduct.\n2. Liability cap — uncapped IP/confidentiality carve-outs asymmetric for smaller vendor. Negotiate separate 3x fees cap.\n3. Force majeure — add pandemic, government closure language plus 5-day notice requirement.",
        "claude-sonnet-4-20250514": "Priority order:\n\n§8.2 is most urgent — Delaware courts read 'arising out of or relating to' to include indemnitee's own negligence. Narrow to material breach, gross negligence, wilful misconduct.\n\nLiability cap: uncapped IP and confidentiality carve-outs neutralise the cap for the scenarios that matter. Separate 3x annual fees cap is reasonable middle ground.\n\nForce majeure: courts have split on pandemic coverage under standard boilerplate — add explicit language now.",
    },
    "Financial Analyst": {
        "gemini-2.0-flash": "3% TGR at 1.2% average revenue growth inflates valuation 30-40%. Run three scenarios: 1%, 2%, 3%. Recheck WACC — use current 10Y treasury, not 5-year average.",
        "gpt-4o": "Two structural issues:\n1. Comp set includes EBITDA-negative companies — remove them. Your 12.4x median shifts to 10.1-10.8x.\n2. Three comps completed acquisitions — adjust for pro-forma revenue.\n\nSensitivity table: ±50bps WACC and ±0.5% growth is too narrow. Expand to ±150bps and ±1.5%.",
        "claude-sonnet-4-20250514": "Sensitivity table ranges are too narrow for current macro. ±50bps WACC, ±0.5% growth won't capture meaningful downside. Expand to ±150bps and ±1.5%.\n\nAdd a second table: revenue growth vs EBITDA margin — that's what management will stress-test.\n\nOn DCF: 3% TGR at 1.2% average growth is hard to defend. Show the 1% scenario prominently as base case.",
    },
}

def get_mock(profession: str, model_id: str) -> str:
    data = MOCK_RESPONSES.get(profession, MOCK_RESPONSES.get("Software Engineer", {}))
    return data.get(model_id, f"[Mock response for {profession} — in production this calls {MODEL_REGISTRY.get(model_id,{}).get('display',model_id)}]")

# ── LinkedIn scraper ────────────────────────────────────────────────────────

async def scrape_linkedin(url: str) -> dict:
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            ctx = await browser.new_context(user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"))
            page = await ctx.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            await asyncio.sleep(2)
            raw = await page.inner_text("body")
            await browser.close()
            lines = [l.strip() for l in raw.split("\n") if l.strip() and len(l.strip()) > 2]
            name = next((l for l in lines[:15] if len(l) > 3 and not any(c in l for c in ["·","|","@","http","Sign","Log"])), None)
            headline = None
            if name and name in lines:
                idx = lines.index(name)
                if idx + 1 < len(lines): headline = lines[idx+1]
            role, company = None, None
            for line in lines:
                m = re.search(r"(.+?)\s+(?:at|@)\s+(.+)", line, re.I)
                if m and not role: role, company = m.group(1).strip(), m.group(2).strip()
            return {"name":name,"headline":headline,"current_role":role or headline,"company":company,"raw_excerpt":raw[:2000],"scrape_success":True,"scrape_method":"playwright"}
    except ImportError:
        return {"scrape_success":False,"scrape_error":"Playwright not installed","scrape_method":"none"}
    except Exception as e:
        return {"scrape_success":False,"scrape_error":str(e)[:200],"scrape_method":"none"}

def mock_linkedin(profession: str) -> dict:
    return {"name":"Alex Chen","headline":f"Senior {profession} · ex-Google","current_role":f"Senior {profession}","company":"TechCorp","scrape_success":True,"scrape_method":"mock"}

# ── Request models ─────────────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    linkedin_url: str
    claimed_profession: str
    submitted_task: str
    use_mock: bool = True

class TasteTestRequest(BaseModel):
    task: str
    profession: str
    session_id: Optional[str] = None
    use_mock: bool = True

class VoteRequest(BaseModel):
    session_id: str
    chosen_index: int

class DimensionRatingRequest(BaseModel):
    session_id: str
    ratings: dict

# ── Verification ────────────────────────────────────────────────────────────

VERIFY_PROMPT = """You verify professional identities for Vantage of AI.

Profile: {profile}
Claimed profession: {profession}
Task (PII masked): {task}

Does this task match a real {profession}'s work in vocabulary and complexity?

Respond ONLY with valid JSON, no markdown:
{{"verdict":"verified","confidence":0.88,"reasoning":"2 sentences","flags":[]}}

verdict: "verified" (>0.75), "review" (0.45-0.75), "flagged" (<0.45)"""

def mock_verify(profession: str, task: str) -> dict:
    words = set(profession.lower().split())
    match = sum(1 for w in words if w in task.lower())
    wc = len(task.split())
    if match >= 1 and wc >= 15:
        return {"verdict":"verified","confidence":0.88,"reasoning":f"Task vocabulary aligns with {profession} work. Domain framing is consistent.","flags":[]}
    elif wc >= 10:
        return {"verdict":"review","confidence":0.60,"reasoning":"Task is plausible but vocabulary is generic.","flags":["Task vocabulary could apply to multiple professions"]}
    return {"verdict":"flagged","confidence":0.28,"reasoning":"Task too short or doesn't match profession.","flags":["Task too short"]}

async def gemini_verify(profession: str, task: str, profile: dict) -> dict:
    try:
        from google import genai as google_genai
        client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        profile_str = f"Name: {profile.get('name','?')} | Role: {profile.get('current_role','?')} | Company: {profile.get('company','?')}"
        prompt = VERIFY_PROMPT.format(profile=profile_str, profession=profession, task=task)
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt))
        text = resp.text.strip()
        if "```" in text: text = text.split("```")[1].lstrip("json").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return mock_verify(profession, task)
    except Exception as e:
        raise HTTPException(500, f"Verify error: {e}")

# ── Taste test ──────────────────────────────────────────────────────────────

async def run_taste_test(task_masked: str, profession: str, use_mock: bool) -> list:
    prof = PROFESSIONS.get(profession, {})
    persona = prof.get("prompt_persona", f"You are an expert {profession}.")
    system = get_system_prompt(profession, persona)
    user = USER_TEMPLATE.replace("{task}", task_masked)

    models_to_use = [m for m in DEFAULT_MODELS]
    random.shuffle(models_to_use)
    labels = ["Model A", "Model B", "Model C"]

    async def one(model_id: str, label: str) -> dict:
        info = MODEL_REGISTRY.get(model_id, {})
        if use_mock or not is_available(model_id):
            response = get_mock(profession, model_id)
            latency_ms = random.uniform(800, 2200) if use_mock else 0
            source = "mock"
        else:
            response, latency_ms = await call_model_async(model_id, system, user)
            source = "live"
        # Run bias check on response
        bias = check_bias(response)
        return {
            "label": label,
            "model": {**info, "id": model_id, "label": label, "source": source},
            "response": response,
            "latency_ms": round(latency_ms, 1),
            "bias": bias,
        }

    results = await asyncio.gather(*[one(models_to_use[i], labels[i]) for i in range(3)])
    return list(results)

async def classify_task(task: str, profession: str) -> str:
    try:
        types = PROFESSIONS[profession]["task_types"]
        from google import genai as google_genai
        client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Classify this {profession} task into exactly one: {', '.join(types)}\n\nTask: {task}\n\nRespond with ONLY the category name."))
        result = resp.text.strip()
        return result if result in types else types[0]
    except:
        return PROFESSIONS.get(profession, {}).get("task_types", ["General"])[0]

# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service":"Vantage of AI","version":"6.0.0",
            "architecture":"async-native + Presidio PII + vertical templates + bias checker + latency tracking",
            "models":available_models(),"professions":list(PROFESSIONS.keys())}

@app.get("/health")
def health():
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        voted = conn.execute("SELECT COUNT(*) FROM sessions WHERE vote IS NOT NULL").fetchone()[0]
    return {"status":"ok","models_live":{MODEL_REGISTRY[m]["display"]:is_available(m) for m in MODEL_REGISTRY},
            "total_sessions":total,"completed_votes":voted,"timestamp":datetime.utcnow().isoformat()}

@app.get("/professions")
def get_professions():
    return {name:{"dimensions":cfg["dimensions"],"task_types":cfg["task_types"],
                  "group":cfg["group"],"credential_label":cfg.get("credential_label","")}
            for name, cfg in PROFESSIONS.items()}

@app.get("/models")
def get_models_route():
    return available_models()

@app.get("/stats")
def stats():
    with get_db() as conn:
        total_votes = conn.execute("SELECT COUNT(*) FROM sessions WHERE vote IS NOT NULL").fetchone()[0]
        total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        by_profession = conn.execute("SELECT profession, COUNT(*) as count FROM sessions WHERE vote IS NOT NULL GROUP BY profession ORDER BY count DESC").fetchall()
        by_model = conn.execute("SELECT winner_model, COUNT(*) as wins, AVG(winner_latency_ms) as avg_latency FROM sessions WHERE vote IS NOT NULL AND winner_model IS NOT NULL GROUP BY winner_model ORDER BY wins DESC").fetchall()
    return {
        "total_votes":total_votes,"total_sessions":total_sessions,
        "by_profession":[{"profession":r["profession"],"count":r["count"]} for r in by_profession],
        "by_model":[{"model":r["winner_model"],"wins":r["wins"],"avg_latency_ms":round(r["avg_latency"] or 0,1)} for r in by_model],
        "generated_at":datetime.utcnow().isoformat(),
    }

@app.get("/session/{session_id}")
def get_session(session_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row: raise HTTPException(404, "Session not found")
        dim_ratings = conn.execute("SELECT dimension, score FROM dimension_ratings WHERE session_id=?", (session_id,)).fetchall()
    if row["vote"] is None: raise HTTPException(400, "Session not yet voted")
    responses = json.loads(row["responses"])
    reveal = [{"label":r["label"],"model_name":r["model"]["display"],"provider":r["model"]["provider"],
               "color":r["model"].get("color","#888"),"latency_ms":r.get("latency_ms",0),"won":i==row["vote"]}
              for i, r in enumerate(responses)]
    return {"session_id":session_id,"profession":row["profession"],"task_type":row["task_type"],
            "winner_model":row["winner_model"],"winner_latency_ms":row["winner_latency_ms"],
            "voted_at":row["voted_at"],"reveal":reveal,"pci_score":row["pci_score"],
            "dimension_ratings":{r["dimension"]:r["score"] for r in dim_ratings},
            "bias_flags":json.loads(row["bias_flags"] or "[]"),
            "share_url":f"https://vantage-ai-txio.onrender.com/scorecard/{session_id}",
            "share_text":f"I blind-tested GPT-4o vs Claude vs Gemini on a real {row['profession']} task — {row['winner_model']} won. No brand bias. Try it: vantage-ai-txio.onrender.com"}

@app.post("/verify")
async def verify(req: VerifyRequest):
    if not req.linkedin_url.startswith("https://www.linkedin.com/in/"):
        raise HTTPException(400, "URL must start with https://www.linkedin.com/in/")
    if len(req.submitted_task.strip()) < 20:
        raise HTTPException(400, "Task must be at least 20 characters")
    if req.claimed_profession not in PROFESSIONS:
        raise HTTPException(400, f"Unknown profession")

    task_masked, pii_count = mask_pii(req.submitted_task)
    profile = mock_linkedin(req.claimed_profession) if req.use_mock else await scrape_linkedin(req.linkedin_url)
    result = mock_verify(req.claimed_profession, task_masked) if req.use_mock else await gemini_verify(req.claimed_profession, task_masked, profile)

    return {**result,
            "trust_score":round(0.5+result["confidence"]*0.2,2),
            "vote_weight":0.5,
            "dimensions":PROFESSIONS[req.claimed_profession]["dimensions"],
            "credential_label":PROFESSIONS[req.claimed_profession].get("credential_label",""),
            "profile":profile,
            "pii_masked":pii_count>0,"pii_count":pii_count,
            "timestamp":datetime.utcnow().isoformat()}

@app.post("/taste-test")
async def taste_test(req: TasteTestRequest):
    if len(req.task.strip()) < 20: raise HTTPException(400, "Task must be at least 20 characters")
    if req.profession not in PROFESSIONS: raise HTTPException(400, "Unknown profession")

    session_id = req.session_id or str(uuid.uuid4())
    task_masked, pii_count = mask_pii(req.task)

    task_type = random.choice(PROFESSIONS[req.profession]["task_types"]) if req.use_mock \
                else await classify_task(task_masked, req.profession)

    responses = await run_taste_test(task_masked, req.profession, req.use_mock)
    blind = [{"label":r["label"],"response":r["response"],"latency_ms":r["latency_ms"]} for r in responses]

    # Aggregate bias flags across all responses
    all_bias_flags = [flag for r in responses for flag in r.get("bias",{}).get("flags",[])]

    with get_db() as conn:
        conn.execute("""INSERT OR REPLACE INTO sessions
            (id,profession,task,task_masked,task_type,responses,vote,pii_count,bias_flags,created_at)
            VALUES (?,?,?,?,?,?,NULL,?,?,?)""",
            (session_id,req.profession,req.task,task_masked,task_type,
             json.dumps(responses),pii_count,json.dumps(all_bias_flags),datetime.utcnow().isoformat()))

    return {"session_id":session_id,"responses":blind,"task_type":task_type,
            "profession_dimensions":PROFESSIONS[req.profession]["dimensions"],
            "pii_masked":pii_count>0,"pii_count":pii_count,
            "bias_flags":all_bias_flags,
            "model_status":[{"display":MODEL_REGISTRY[m]["display"],"provider":MODEL_REGISTRY[m]["provider"],
                              "live":is_available(m),"color":MODEL_REGISTRY[m]["color"]} for m in MODEL_REGISTRY]}

@app.post("/vote")
def vote(req: VoteRequest):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (req.session_id,)).fetchone()
        if not row: raise HTTPException(404, "Session not found")
        if row["vote"] is not None: raise HTTPException(400, "Already voted")
        if req.chosen_index not in [0,1,2]: raise HTTPException(400, "chosen_index must be 0-2")

        responses = json.loads(row["responses"])
        winner = responses[req.chosen_index]
        winner_model = winner["model"]["display"]
        winner_color = winner["model"].get("color","#4285F4")
        winner_latency = winner.get("latency_ms", 0)

        total = conn.execute("SELECT COUNT(*) FROM sessions WHERE profession=? AND vote IS NOT NULL",(row["profession"],)).fetchone()[0]+1
        same = conn.execute("SELECT COUNT(*) FROM sessions WHERE profession=? AND winner_model=? AND vote IS NOT NULL",(row["profession"],winner_model)).fetchone()[0]+1
        pci = compute_pci(winner_model, row["profession"], same, total)

        conn.execute("UPDATE sessions SET vote=?,winner_model=?,winner_latency_ms=?,voted_at=?,pci_score=? WHERE id=?",
                     (req.chosen_index,winner_model,winner_latency,datetime.utcnow().isoformat(),pci,req.session_id))

    peer_pct = round((same/total)*100) if total>2 else None
    reveal = [{"label":r["label"],"model_name":r["model"]["display"],"provider":r["model"]["provider"],
               "color":r["model"].get("color","#888"),"response":r["response"],
               "latency_ms":r.get("latency_ms",0),"won":i==req.chosen_index,
               "source":r["model"].get("source","mock")} for i,r in enumerate(responses)]

    return {
        "session_id":req.session_id,
        "chosen_index":req.chosen_index,
        "winner_label":winner["label"],
        "winner_model":winner_model,
        "winner_color":winner_color,
        "winner_latency_ms":winner_latency,
        "pci_score":pci,
        "surprise_message":random.choice([
            f"You preferred {winner['label']} — that was {winner_model}.",
            f"Your pick was {winner['label']} — {winner_model}.",
            f"{winner_model} won your vote as {winner['label']}.",
        ]),
        "peer_comparison":f"{peer_pct}% of {row['profession']}s preferred {winner_model}" if peer_pct else None,
        "reveal":reveal,
        "dimensions":PROFESSIONS[row["profession"]]["dimensions"],
        "scorecard":{
            "profession":row["profession"],"task_type":row["task_type"],
            "task_preview":row["task"][:90]+"...","winner":winner_model,
            "winner_color":winner_color,"pci_score":pci,
            "winner_latency_ms":winner_latency,
            "share_url":f"https://vantage-ai-txio.onrender.com/scorecard/{req.session_id}",
            "share_text":f"I blind-tested GPT-4o vs Claude vs Gemini on a real {row['profession']} task — {winner_model} won (PCI: {pci:.2f}). No brand bias. Try it: vantage-ai-txio.onrender.com/scorecard/{req.session_id}",
        }
    }

@app.post("/rate-dimensions")
def rate_dimensions(req: DimensionRatingRequest):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (req.session_id,)).fetchone()
        if not row: raise HTTPException(404, "Session not found")
        if row["vote"] is None: raise HTTPException(400, "Vote first")
        for dim, score in req.ratings.items():
            if not (1<=score<=5): raise HTTPException(400, "Score must be 1-5")
            conn.execute("INSERT INTO dimension_ratings (session_id,profession,winner_model,dimension,score,created_at) VALUES (?,?,?,?,?,?)",
                (req.session_id,row["profession"],row["winner_model"],dim,score,datetime.utcnow().isoformat()))
        conn.execute("UPDATE sessions SET dimension_scores=? WHERE id=?", (json.dumps(req.ratings),req.session_id))
    return {"status":"ok","ratings_saved":len(req.ratings)}

@app.get("/leaderboard")
def leaderboard(profession: Optional[str] = None):
    with get_db() as conn:
        if profession:
            rows = conn.execute("SELECT winner_model,task_type,COUNT(*) as votes,AVG(winner_latency_ms) as avg_lat FROM sessions WHERE vote IS NOT NULL AND profession=? GROUP BY winner_model,task_type ORDER BY votes DESC",(profession,)).fetchall()
        else:
            rows = conn.execute("SELECT winner_model,profession,COUNT(*) as votes,AVG(winner_latency_ms) as avg_lat FROM sessions WHERE vote IS NOT NULL GROUP BY winner_model,profession ORDER BY votes DESC").fetchall()

        dim_rows = conn.execute("SELECT winner_model,dimension,AVG(score) as avg_score,COUNT(*) as n FROM dimension_ratings GROUP BY winner_model,dimension ORDER BY avg_score DESC").fetchall()
        total_votes = conn.execute("SELECT COUNT(*) FROM sessions WHERE vote IS NOT NULL").fetchone()[0]
        total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        prof_summary = conn.execute("SELECT profession,winner_model,COUNT(*) as votes FROM sessions WHERE vote IS NOT NULL GROUP BY profession,winner_model ORDER BY profession,votes DESC").fetchall()

    model_map = {v["display"]:{"id":k,**v} for k,v in MODEL_REGISTRY.items()}
    model_votes: dict = {}
    for r in rows:
        m = r["winner_model"]
        if m not in model_votes:
            info = model_map.get(m,{})
            wins = sum(rr["votes"] for rr in rows if rr["winner_model"]==m)
            pci = compute_pci(m, profession or "Software Engineer", wins, total_sessions)
            model_votes[m] = {"model":m,"provider":info.get("provider",""),"color":info.get("color","#888"),
                               "total_votes":0,"pci_score":pci,"avg_latency_ms":0,
                               "by_profession":{},"by_task_type":{}}
        model_votes[m]["total_votes"] += r["votes"]
        if r["avg_lat"]: model_votes[m]["avg_latency_ms"] = round(r["avg_lat"],1)
        key = "by_task_type" if profession else "by_profession"
        sub = r["task_type"] if profession else r["profession"]
        model_votes[m][key][sub] = r["votes"]

    rankings = sorted(model_votes.values(), key=lambda x:(x["pci_score"],x["total_votes"]), reverse=True)
    for i,r in enumerate(rankings): r["rank"] = i+1

    dim_scores: dict = {}
    for r in dim_rows:
        m = r["winner_model"]
        if m not in dim_scores: dim_scores[m] = {}
        dim_scores[m][r["dimension"]] = {"avg":round(r["avg_score"],2),"n":r["n"]}

    prof_breakdown: dict = {}
    for r in prof_summary:
        if r["profession"] not in prof_breakdown: prof_breakdown[r["profession"]] = []
        prof_breakdown[r["profession"]].append({"model":r["winner_model"],"votes":r["votes"]})

    return {"total_votes":total_votes,"total_sessions":total_sessions,
            "rankings":rankings,"dimension_scores":dim_scores,
            "by_profession":prof_breakdown,"generated_at":datetime.utcnow().isoformat()}

@app.get("/results")
def results():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM sessions WHERE vote IS NOT NULL ORDER BY voted_at DESC LIMIT 50").fetchall()
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        voted = conn.execute("SELECT COUNT(*) FROM sessions WHERE vote IS NOT NULL").fetchone()[0]
    return {"total_sessions":total,"completed":voted,"sessions":[dict(r) for r in rows]}
