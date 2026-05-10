"""
Vantage of AI — Backend v9
===========================
New in v9:
  - RAG pipeline — task enriched with retrieved professional context
  - File upload — any file type, extracted text injected into model prompt
  - /models page data endpoint — SEO content for public models page
  - Improved taste-test with file context mode (context vs reference)

Install:
  pip install fastapi uvicorn google-genai openai anthropic python-dotenv
      passlib[bcrypt] python-jose[cryptography] httpx Pillow PyPDF2
      python-docx openpyxl

Run:
  uvicorn main:app --reload --port 8000
"""

import os, json, asyncio, uuid, random, sqlite3, re, time, logging, math, hashlib, base64
from datetime import datetime, timedelta
from typing import Optional
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger("vantage")

# Import RAG pipeline and file processor
try:
    from rag import rag
    from file_processor import extract_text, build_file_context
    logger.info("✓ RAG pipeline and file processor loaded")
except ImportError as e:
    logger.warning("RAG/file processor not available: %s", e)
    rag = None
    def extract_text(b, f, m=""): return "", "unavailable"
    def build_file_context(t, f, m): return f"[File: {f}]\n{t}\n"

app = FastAPI(title="Vantage of AI", version="9.0.0")

# CORS must be added FIRST before any other middleware
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Serve frontend static files
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Request
import pathlib

# Mount frontend if it exists (when running from repo root)
_frontend_path = pathlib.Path(__file__).parent.parent / "frontend"
if _frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_path)), name="static")

# Explicit OPTIONS handler for preflight requests
@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "*",
        }
    )

# ── Config ─────────────────────────────────────────────────────────────────

JWT_SECRET = os.getenv("JWT_SECRET", "vantage-secret-change-in-production-" + uuid.uuid4().hex)
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24       # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = 30
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

# ── SQLite ─────────────────────────────────────────────────────────────────

DB_PATH = "vantage.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            name TEXT,
            avatar_url TEXT,
            password_hash TEXT,
            google_id TEXT UNIQUE,
            profession TEXT,
            trust_score REAL DEFAULT 0.5,
            vote_count INTEGER DEFAULT 0,
            created_at TEXT,
            last_login TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT,
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
            voted_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS dimension_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_id TEXT,
            profession TEXT,
            winner_model TEXT,
            dimension TEXT,
            score INTEGER,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS refresh_tokens (
            token TEXT PRIMARY KEY,
            user_id TEXT,
            expires_at TEXT,
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

# ── Auth helpers ────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    try:
        import bcrypt
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    except ImportError:
        return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain: str, hashed: str) -> bool:
    try:
        import bcrypt
        return bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))
    except ImportError:
        return hashlib.sha256(plain.encode()).hexdigest() == hashed

def create_access_token(user_id: str, email: str) -> str:
    from jose import jwt as jose_jwt
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jose_jwt.encode(
        {"sub": user_id, "email": email, "exp": expire, "type": "access"},
        JWT_SECRET, algorithm=JWT_ALGORITHM
    )

def create_refresh_token(user_id: str) -> str:
    token = uuid.uuid4().hex + uuid.uuid4().hex
    expires = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO refresh_tokens (token, user_id, expires_at, created_at) VALUES (?,?,?,?)",
            (token, user_id, expires.isoformat(), datetime.utcnow().isoformat())
        )
    return token

def verify_access_token(token: str) -> Optional[dict]:
    try:
        from jose import jwt as jose_jwt, JWTError
        payload = jose_jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            return None
        return payload
    except Exception:
        return None

security = HTTPBearer(auto_error=False)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[dict]:
    if not credentials:
        return None
    payload = verify_access_token(credentials.credentials)
    if not payload:
        return None
    with get_db() as conn:
        user = conn.execute("SELECT * FROM users WHERE id=?", (payload["sub"],)).fetchone()
    return dict(user) if user else None

def require_auth(user = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user

# ── Model registry ──────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "gemini-2.0-flash":        {"display":"Gemini 2.0 Flash","provider":"Google",    "env_key":"GEMINI_API_KEY",    "color":"#4285F4"},
    "gpt-4o":                  {"display":"GPT-4o",          "provider":"OpenAI",    "env_key":"OPENAI_API_KEY",    "color":"#10A37F"},
    "claude-sonnet-4-20250514":{"display":"Claude Sonnet",   "provider":"Anthropic", "env_key":"ANTHROPIC_API_KEY", "color":"#D97757"},
}
DEFAULT_MODELS = list(MODEL_REGISTRY.keys())

def is_available(model_id: str) -> bool:
    return bool(MODEL_REGISTRY.get(model_id) and os.getenv(MODEL_REGISTRY[model_id]["env_key"]))

def available_models():
    return [{"id":k,**v,"live":is_available(k)} for k,v in MODEL_REGISTRY.items()]

# ── Professions ─────────────────────────────────────────────────────────────

PROFESSIONS = {
    "Contract Attorney":     {"group":"Professional","dimensions":["Legal accuracy","Risk identification","Practical advice"],"task_types":["Contract review","Risk assessment","Negotiation strategy","Compliance","Dispute resolution"],"prompt_persona":"You are a senior contract attorney with 12 years of corporate law experience."},
    "Radiologist":           {"group":"Professional","dimensions":["Clinical accuracy","Differential completeness","Safety awareness"],"task_types":["Image interpretation","Differential diagnosis","Report writing","Protocol selection","Findings correlation"],"prompt_persona":"You are a board-certified radiologist with subspecialty training in diagnostic imaging."},
    "Financial Analyst":     {"group":"Professional","dimensions":["Analytical rigor","Model accuracy","Business insight"],"task_types":["Valuation","Financial modeling","Due diligence","Market analysis","Risk modeling"],"prompt_persona":"You are a CFA-certified senior financial analyst at a top-tier investment bank."},
    "Accountant / CPA":      {"group":"Professional","dimensions":["Regulatory accuracy","Practical application","Risk flagging"],"task_types":["Tax planning","Financial reporting","Audit prep","Compliance","Advisory"],"prompt_persona":"You are a CPA with 10 years in corporate accounting and tax advisory."},
    "HR Manager":            {"group":"Professional","dimensions":["Policy correctness","Empathy & tone","Actionability"],"task_types":["Policy writing","Performance management","Conflict resolution","Hiring & onboarding","Compensation"],"prompt_persona":"You are an experienced HR manager who built people ops at two high-growth startups."},
    "Journalist":            {"group":"Professional","dimensions":["Factual accuracy","Source quality","Narrative clarity"],"task_types":["Story angle","Research","Headline writing","Interview prep","Fact-checking"],"prompt_persona":"You are an investigative journalist with 10 years at a national publication."},
    "Nurse Practitioner":    {"group":"Professional","dimensions":["Clinical accuracy","Patient safety","Evidence-based reasoning"],"task_types":["Patient assessment","Treatment planning","Medication management","Documentation","Patient education"],"prompt_persona":"You are a board-certified nurse practitioner with 8 years in primary care."},
    "Software Engineer":     {"group":"Technical","dimensions":["Technical accuracy","Code clarity","Scalability thinking"],"task_types":["System design","Code review","Debugging","Architecture","Performance"],"prompt_persona":"You are a senior software engineer with deep expertise in distributed systems and production engineering."},
    "ML Engineer":           {"group":"Technical","dimensions":["Model correctness","Production readiness","Efficiency"],"task_types":["Model optimization","Pipeline design","Serving & inference","Monitoring","Training infrastructure"],"prompt_persona":"You are a senior ML engineer who builds and ships production ML systems at scale."},
    "Data Scientist":        {"group":"Technical","dimensions":["Statistical correctness","Code quality","Assumption clarity"],"task_types":["Model selection","EDA & visualization","Feature engineering","Evaluation & metrics","Deployment & MLOps"],"prompt_persona":"You are a senior data scientist with 8 years in ML and statistical modeling."},
    "DevOps / Platform Eng": {"group":"Technical","dimensions":["Infrastructure correctness","Security awareness","Reliability thinking"],"task_types":["CI/CD pipelines","Cloud architecture","Incident response","Cost optimisation","Security hardening"],"prompt_persona":"You are a senior platform engineer with expertise in Kubernetes, Terraform, and cloud-native infrastructure."},
    "Cybersecurity Analyst": {"group":"Technical","dimensions":["Threat accuracy","Remediation quality","Risk communication"],"task_types":["Threat modeling","Incident response","Vulnerability assessment","Policy writing","Penetration testing"],"prompt_persona":"You are a senior cybersecurity analyst with expertise in threat detection and incident response."},
    "Data Engineer":         {"group":"Technical","dimensions":["Pipeline correctness","Data quality","Scalability"],"task_types":["Pipeline design","Data modeling","ETL/ELT","Data quality","Query optimisation"],"prompt_persona":"You are a senior data engineer who builds and maintains large-scale data pipelines."},
    "UX Designer":           {"group":"Creative","dimensions":["User empathy","Design rationale","Feasibility"],"task_types":["User research","Wireframing","Usability review","Design systems","Accessibility"],"prompt_persona":"You are a senior UX designer with a portfolio spanning fintech, healthtech, and consumer apps."},
    "Marketing Manager":     {"group":"Creative","dimensions":["Strategic clarity","Audience targeting","Measurable outcomes"],"task_types":["Campaign strategy","Copy & messaging","Channel selection","Analytics & attribution","Brand positioning"],"prompt_persona":"You are a senior marketing manager who has led growth at multiple B2B SaaS companies."},
    "Copywriter":            {"group":"Creative","dimensions":["Persuasiveness","Brand alignment","Originality"],"task_types":["Ad copy","Email campaigns","Landing pages","Social content","Brand voice"],"prompt_persona":"You are a senior copywriter with 8 years across DTC, SaaS, and agency work."},
    "Product Manager":       {"group":"Operational","dimensions":["User focus","Prioritization logic","Measurability"],"task_types":["PRD writing","Prioritization","Metrics & OKRs","User research","Roadmapping"],"prompt_persona":"You are a senior product manager with a track record of 0→1 products at Series B companies."},
    "Operations Manager":    {"group":"Operational","dimensions":["Process clarity","Efficiency focus","Risk awareness"],"task_types":["Process design","KPI tracking","Vendor management","Cost reduction","Team coordination"],"prompt_persona":"You are a senior operations manager who has scaled processes from startup to growth stage."},
    "Supply Chain Analyst":  {"group":"Operational","dimensions":["Quantitative accuracy","Risk identification","Practical recommendations"],"task_types":["Demand forecasting","Inventory optimisation","Supplier analysis","Logistics","Risk mitigation"],"prompt_persona":"You are a senior supply chain analyst with expertise in forecasting, inventory, and logistics."},
    "Real Estate Analyst":   {"group":"Operational","dimensions":["Market accuracy","Financial rigor","Risk assessment"],"task_types":["Property valuation","Market analysis","Investment modeling","Due diligence","Portfolio analysis"],"prompt_persona":"You are a senior real estate analyst specialising in commercial property valuation."},
    "Customer Success Mgr":  {"group":"Operational","dimensions":["Empathy & tone","Problem-solving","Retention focus"],"task_types":["Churn prevention","Onboarding","QBR preparation","Escalation handling","Expansion planning"],"prompt_persona":"You are a senior customer success manager managing enterprise accounts at a SaaS company."},
    "Executive Assistant":   {"group":"Operational","dimensions":["Clarity","Professionalism","Attention to detail"],"task_types":["Email drafting","Meeting prep","Travel coordination","Document management","Stakeholder communication"],"prompt_persona":"You are a senior executive assistant supporting C-suite leaders at a Fortune 500 company."},
}

# ── PII masking ─────────────────────────────────────────────────────────────

_PII_PATTERNS = [
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL]'),
    (re.compile(r'\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'), '[PHONE]'),
    (re.compile(r'https?://\S+|www\.\S+'), '[URL]'),
    (re.compile(r'\b(Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?'), '[NAME]'),
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN]'),
]

def mask_pii(text: str) -> tuple[str, int]:
    masked, count = text, 0
    for pattern, replacement in _PII_PATTERNS:
        new = pattern.sub(replacement, masked)
        count += len(pattern.findall(masked))
        masked = new
    return masked, count

# ── Bias checker ─────────────────────────────────────────────────────────────

_MALE = re.compile(r"\b(he|him|his|himself)\b", re.I)
_FEMALE = re.compile(r"\b(she|her|hers|herself)\b", re.I)
_NEG = re.compile(r"\b(weak|incompetent|unqualified|inferior|risky|problematic)\b", re.I)
_DEMO = ["men","women","male","female","elderly","young","minority","white","black","asian","hispanic"]

def check_bias(text: str) -> list:
    flags = []
    m, f = len(_MALE.findall(text)), len(_FEMALE.findall(text))
    if m + f > 4:
        ratio = m / (m + f + 1e-6)
        if ratio > 0.85: flags.append("Predominantly male pronouns")
        elif ratio < 0.15: flags.append("Predominantly female pronouns")
    tl = text.lower()
    for g in _DEMO:
        if g in tl:
            idx = tl.find(g)
            window = tl[max(0,idx-50):idx+len(g)+50]
            if len(_NEG.findall(window)) > 1:
                flags.append(f"Negative sentiment near '{g}'")
    return flags

# ── Vertical prompt templates ────────────────────────────────────────────────

VERTICAL_TEMPLATES = {
    "Contract Attorney": "You are a legal AI assistant evaluated by verified legal professionals.\n\nBase analysis ONLY on provided context. Cite specific clauses. Flag jurisdictional assumptions. Do not speculate beyond evidence.",
    "Radiologist": "You are a clinical AI assistant evaluated by verified healthcare professionals.\n\nBase response on clinical evidence only. Flag uncertainty with [LOW CONFIDENCE]. Never fabricate clinical facts.",
    "Financial Analyst": "You are a financial AI assistant evaluated by verified finance professionals.\n\nGround all analysis in provided data. Cite specific figures. Flag extrapolations clearly. Apply GAAP/IFRS explicitly.",
    "Software Engineer": "You are a software engineering AI assistant evaluated by verified engineers.\n\nProvide specific, actionable guidance. Explain reasoning. Flag trade-offs explicitly (performance vs maintainability).",
    "ML Engineer": "You are an ML engineering AI assistant evaluated by verified ML engineers.\n\nGround recommendations in production ML best practices. Flag latency, memory, scalability implications.",
    "Data Scientist": "You are a data science AI assistant evaluated by verified data scientists.\n\nGround statistical recommendations in rigorous methodology. Flag assumptions. Distinguish correlation from causation.",
    "HR Manager": "You are an HR AI assistant evaluated by HR professionals.\n\nApply employment law principles for the stated jurisdiction. Flag equity or compliance concerns. Treat all data as sensitive.",
    "Product Manager": "You are a product management AI assistant evaluated by verified PMs.\n\nGround in user research and business context. Flag assumptions about user behaviour. Be specific about metrics.",
}

DEFAULT_TEMPLATE = "You are a domain expert AI assistant evaluated by verified professionals.\n\nBe specific, actionable, and grounded in domain knowledge. Flag uncertainty. Under 200 words."
USER_TEMPLATE = "Professional task:\n{task}\n\nResponse:"

def get_system_prompt(profession: str) -> str:
    persona = PROFESSIONS.get(profession, {}).get("prompt_persona", f"You are an expert {profession}.")
    template = VERTICAL_TEMPLATES.get(profession, DEFAULT_TEMPLATE)
    return f"{persona}\n\n{template}"

# ── Mock responses ───────────────────────────────────────────────────────────

MOCK_RESPONSES = {
    "Data Scientist": {
        "gemini-2.0-flash": "Switch to XGBoost with scale_pos_weight=11.5 first — fastest path to better recall. Lower classification threshold to 0.2–0.3 using PR curve, not ROC. Check SHAP values first — if churn features have low importance, the problem is feature engineering, not the model.",
        "gpt-4o": "Three steps:\n1. Threshold tuning — move from 0.5 to 0.25. Expect recall to jump to 65–70%.\n2. XGBoost with class_weight='balanced', scale_pos_weight ~11.5.\n3. SMOTE only as last resort — synthetic samples don't reflect real churn patterns.\n\nUse stratified k-fold with F1 throughout. Accuracy at 8% imbalance is meaningless.",
        "claude-sonnet-4-20250514": "Before changing models, run SHAP on your logistic regression. If churn signals (session drop, plan downgrade, support tickets) have low importance, no model change will help — fix features first.\n\nIf features look healthy: threshold adjustment to 0.25 is your fastest win, then XGBoost. Switch evaluation metric to F1 or AUC-PR. Accuracy at 8% imbalance tells you nothing.",
    },
    "Software Engineer": {
        "gemini-2.0-flash": "Write-through with Redlock (3 Redis nodes) and 300s TTL ±30s jitter. Circuit breaker for Redis failures. ~2ms latency overhead, stampede risk eliminated.",
        "gpt-4o": "Three options:\n1. Write-through + jitter: 300s ±30s. Eliminates thundering herd.\n2. Cache-aside + optimistic locking: compare-and-set with version tag.\n3. Redlock: correct but operationally heavier.\n\n60s TTL is too aggressive — ~1.6M DB hits/day. Bump to 600s, invalidate on writes.",
        "claude-sonnet-4-20250514": "Stampede and race condition are separate problems. TTL jitter (±30s on 300s) covers 90% of stampede cases. Cache-aside with optimistic locking handles the race condition. Redlock only if both fail — it's correct but adds operational complexity.",
    },
    "Contract Attorney": {
        "gemini-2.0-flash": "§8.2 is overbroad — 'arising out of or relating to' covers indemnitee's own negligence under Delaware law. Narrow to: material breach, gross negligence, wilful misconduct. Add contributory negligence carve-out.",
        "gpt-4o": "Three issues:\n1. §8.2 indemnification — too broad. Narrow trigger to material breach, gross negligence, wilful misconduct.\n2. Liability cap — uncapped IP/confidentiality carve-outs asymmetric. Negotiate separate 3x annual fees cap.\n3. Force majeure — add pandemic, government closure language plus 5-day notice requirement.",
        "claude-sonnet-4-20250514": "§8.2 is most urgent — Delaware courts read 'arising out of or relating to' to include indemnitee's own negligence. Narrow to material breach, gross negligence, wilful misconduct.\n\nLiability cap: uncapped IP and confidentiality carve-outs neutralise the cap for situations that matter. Separate 3x annual fees cap is reasonable middle ground.",
    },
    "Financial Analyst": {
        "gemini-2.0-flash": "3% TGR at 1.2% average revenue growth inflates valuation 30-40%. Run three scenarios: 1%, 2%, 3%. Recheck WACC — use current 10Y treasury, not 5-year average.",
        "gpt-4o": "Two structural issues:\n1. Comp set includes EBITDA-negative companies — remove them. Your 12.4x median shifts to 10.1-10.8x.\n2. Three comps completed acquisitions — adjust for pro-forma revenue.\n\nSensitivity table: ±50bps WACC and ±0.5% growth is too narrow. Expand to ±150bps and ±1.5%.",
        "claude-sonnet-4-20250514": "Sensitivity table ranges are too narrow. ±50bps WACC and ±0.5% growth won't capture meaningful downside. Expand to ±150bps and ±1.5%.\n\nAdd second table: revenue growth vs EBITDA margin — that's what management stress-tests in board meetings. On DCF: 3% TGR at 1.2% average growth is hard to defend.",
    },
}

def get_mock(profession: str, model_id: str) -> str:
    data = MOCK_RESPONSES.get(profession, MOCK_RESPONSES.get("Software Engineer", {}))
    return data.get(model_id, f"[Expert {profession} response from {MODEL_REGISTRY.get(model_id,{}).get('display',model_id)}]")

# ── Async model caller ────────────────────────────────────────────────────────

_semaphore = asyncio.Semaphore(10)

async def call_model(model_id: str, system: str, user: str) -> tuple[str, float]:
    info = MODEL_REGISTRY.get(model_id, {})
    provider = info.get("provider", "")
    start = time.perf_counter()
    async with _semaphore:
        try:
            if provider == "Google":
                from google import genai as g
                client = g.Client(api_key=os.getenv("GEMINI_API_KEY"))
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(None, lambda: client.models.generate_content(
                    model=model_id, contents=f"{system}\n\n{user}"))
                text = resp.text.strip()
            elif provider == "OpenAI":
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                resp = await client.chat.completions.create(
                    model=model_id, max_tokens=400,
                    messages=[{"role":"system","content":system},{"role":"user","content":user}])
                text = resp.choices[0].message.content.strip()
            elif provider == "Anthropic":
                import anthropic
                client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                msg = await client.messages.create(
                    model=model_id, max_tokens=400,
                    system=[{"type":"text","text":system,"cache_control":{"type":"ephemeral"}}],
                    messages=[{"role":"user","content":user}])
                text = msg.content[0].text.strip()
            else:
                text = f"[Unknown provider: {provider}]"
        except Exception as e:
            logger.error("Model %s error: %s", model_id, e)
            text = f"[{info.get('display',model_id)} temporarily unavailable]"
    return text, (time.perf_counter() - start) * 1000

# ── PCI scoring ───────────────────────────────────────────────────────────────

def wilson_lb(wins: int, total: int) -> float:
    if total == 0: return 0.0
    z, p = 1.96, wins/total
    denom = 1 + z**2/total
    return max(0.0, (p + z**2/(2*total) - z*((p*(1-p)/total + z**2/(4*total**2))**0.5)) / denom)

REF_SCORES = {
    "Gemini 2.0 Flash":{"a":0.72,"k":0.81,"m":0.85,"c":0.80,"s":0.74,"t":0.70},
    "GPT-4o":          {"a":0.79,"k":0.88,"m":0.92,"c":0.88,"s":0.85,"t":0.75},
    "Claude Sonnet":   {"a":0.75,"k":0.85,"m":0.87,"c":0.84,"s":0.80,"t":0.82},
}

def compute_pci(model_display: str, wins: int, total: int) -> float:
    ref = REF_SCORES.get(model_display, {})
    pub = sum(ref.values()) / max(len(ref), 1) * 0.55
    pro = wilson_lb(wins, total) * 0.30
    rigor = min(1.0, total/50) * wilson_lb(wins, total) * 0.15
    return round(min(1.0, pub + pro + rigor), 4)

# ── Request models ────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str
    profession: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class GoogleAuthRequest(BaseModel):
    id_token: str

class RefreshRequest(BaseModel):
    refresh_token: str

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
    file_id: Optional[str] = None        # from /upload endpoint
    file_mode: Optional[str] = "context" # "context" or "reference"

class VoteRequest(BaseModel):
    session_id: str
    chosen_index: int

class DimensionRatingRequest(BaseModel):
    session_id: str
    ratings: dict

# ── Auth routes ────────────────────────────────────────────────────────────────

@app.post("/auth/register")
def register(req: RegisterRequest):
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    user_id = str(uuid.uuid4())
    pw_hash = hash_password(req.password)
    try:
        with get_db() as conn:
            conn.execute("""INSERT INTO users (id,email,name,profession,password_hash,trust_score,vote_count,created_at,last_login)
                VALUES (?,?,?,?,?,0.5,0,?,?)""",
                (user_id, req.email.lower(), req.name, req.profession, pw_hash,
                 datetime.utcnow().isoformat(), datetime.utcnow().isoformat()))
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Email already registered")

    access = create_access_token(user_id, req.email)
    refresh = create_refresh_token(user_id)
    return {"access_token": access, "refresh_token": refresh, "token_type": "bearer",
            "user": {"id": user_id, "email": req.email, "name": req.name,
                     "profession": req.profession, "trust_score": 0.5, "vote_count": 0}}

@app.post("/auth/login")
def login(req: LoginRequest):
    with get_db() as conn:
        user = conn.execute("SELECT * FROM users WHERE email=?", (req.email.lower(),)).fetchone()
    if not user or not user["password_hash"] or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")
    with get_db() as conn:
        conn.execute("UPDATE users SET last_login=? WHERE id=?", (datetime.utcnow().isoformat(), user["id"]))
    access = create_access_token(user["id"], user["email"])
    refresh = create_refresh_token(user["id"])
    return {"access_token": access, "refresh_token": refresh, "token_type": "bearer",
            "user": {"id":user["id"],"email":user["email"],"name":user["name"],
                     "profession":user["profession"],"trust_score":user["trust_score"],"vote_count":user["vote_count"]}}

@app.post("/auth/google")
async def google_auth(req: GoogleAuthRequest):
    """Verify Google ID token and create/login user."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={req.id_token}")
            if resp.status_code != 200:
                raise HTTPException(401, "Invalid Google token")
            info = resp.json()
            if info.get("aud") != GOOGLE_CLIENT_ID and GOOGLE_CLIENT_ID:
                raise HTTPException(401, "Token audience mismatch")

        google_id = info["sub"]
        email = info["email"]
        name = info.get("name", email.split("@")[0])
        avatar = info.get("picture", "")

        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE google_id=? OR email=?", (google_id, email)).fetchone()
            if user:
                conn.execute("UPDATE users SET google_id=?,last_login=?,avatar_url=? WHERE id=?",
                             (google_id, datetime.utcnow().isoformat(), avatar, user["id"]))
                user_id = user["id"]
                vote_count = user["vote_count"]
                trust_score = user["trust_score"]
                profession = user["profession"]
            else:
                user_id = str(uuid.uuid4())
                vote_count, trust_score, profession = 0, 0.5, None
                conn.execute("""INSERT INTO users (id,email,name,avatar_url,google_id,trust_score,vote_count,created_at,last_login)
                    VALUES (?,?,?,?,?,0.5,0,?,?)""",
                    (user_id, email, name, avatar, google_id,
                     datetime.utcnow().isoformat(), datetime.utcnow().isoformat()))

        access = create_access_token(user_id, email)
        refresh = create_refresh_token(user_id)
        return {"access_token": access, "refresh_token": refresh, "token_type": "bearer",
                "user": {"id":user_id,"email":email,"name":name,"avatar_url":avatar,
                         "profession":profession,"trust_score":trust_score,"vote_count":vote_count}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Google auth error: {e}")

@app.post("/auth/refresh")
def refresh_token(req: RefreshRequest):
    with get_db() as conn:
        token_row = conn.execute("SELECT * FROM refresh_tokens WHERE token=?", (req.refresh_token,)).fetchone()
        if not token_row:
            raise HTTPException(401, "Invalid refresh token")
        if datetime.fromisoformat(token_row["expires_at"]) < datetime.utcnow():
            conn.execute("DELETE FROM refresh_tokens WHERE token=?", (req.refresh_token,))
            raise HTTPException(401, "Refresh token expired")
        user = conn.execute("SELECT * FROM users WHERE id=?", (token_row["user_id"],)).fetchone()
        if not user:
            raise HTTPException(401, "User not found")
        conn.execute("DELETE FROM refresh_tokens WHERE token=?", (req.refresh_token,))

    access = create_access_token(user["id"], user["email"])
    new_refresh = create_refresh_token(user["id"])
    return {"access_token": access, "refresh_token": new_refresh}

@app.get("/auth/me")
def get_me(user = Depends(require_auth)):
    return {"id":user["id"],"email":user["email"],"name":user["name"],
            "avatar_url":user.get("avatar_url"),"profession":user.get("profession"),
            "trust_score":user["trust_score"],"vote_count":user["vote_count"],
            "created_at":user["created_at"]}

@app.get("/auth/history")
def get_history(user = Depends(require_auth)):
    with get_db() as conn:
        sessions = conn.execute("""
            SELECT id,profession,task,task_type,winner_model,pci_score,created_at,voted_at
            FROM sessions WHERE user_id=? AND vote IS NOT NULL
            ORDER BY voted_at DESC LIMIT 20
        """, (user["id"],)).fetchall()
    return {"sessions": [dict(s) for s in sessions], "total": len(sessions)}

# ── Core routes ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    # Serve frontend if available
    frontend_index = pathlib.Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_index.exists():
        return FileResponse(str(frontend_index))
    return {"service":"Vantage of AI","version":"9.0.0",
            "models":available_models(),"professions":list(PROFESSIONS.keys())}

@app.get("/app")
def serve_app():
    frontend_index = pathlib.Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_index.exists():
        return FileResponse(str(frontend_index))
    return {"error": "Frontend not found"}

@app.get("/health")
def health():
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        voted = conn.execute("SELECT COUNT(*) FROM sessions WHERE vote IS NOT NULL").fetchone()[0]
        users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    return {"status":"ok","models_live":{MODEL_REGISTRY[m]["display"]:is_available(m) for m in MODEL_REGISTRY},
            "total_sessions":total,"completed_votes":voted,"registered_users":users,
            "timestamp":datetime.utcnow().isoformat()}

@app.get("/professions")
def get_professions():
    return {name:{"dimensions":cfg["dimensions"],"task_types":cfg["task_types"],"group":cfg["group"]}
            for name, cfg in PROFESSIONS.items()}

@app.get("/stats")
def stats():
    with get_db() as conn:
        total_votes = conn.execute("SELECT COUNT(*) FROM sessions WHERE vote IS NOT NULL").fetchone()[0]
        total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        by_profession = conn.execute("""SELECT profession, COUNT(*) as count FROM sessions
            WHERE vote IS NOT NULL GROUP BY profession ORDER BY count DESC""").fetchall()
        by_model = conn.execute("""SELECT winner_model, COUNT(*) as wins FROM sessions
            WHERE vote IS NOT NULL AND winner_model IS NOT NULL
            GROUP BY winner_model ORDER BY wins DESC""").fetchall()
    return {"total_votes":total_votes,"total_sessions":total_sessions,"total_users":total_users,
            "by_profession":[{"profession":r["profession"],"count":r["count"]} for r in by_profession],
            "by_model":[{"model":r["winner_model"],"wins":r["wins"]} for r in by_model],
            "generated_at":datetime.utcnow().isoformat()}

@app.post("/verify")
async def verify(req: VerifyRequest, user = Depends(get_current_user)):
    if not req.linkedin_url.startswith("https://www.linkedin.com/in/"):
        raise HTTPException(400, "URL must start with https://www.linkedin.com/in/")
    if len(req.submitted_task.strip()) < 20:
        raise HTTPException(400, "Task must be at least 20 characters")
    if req.claimed_profession not in PROFESSIONS:
        raise HTTPException(400, "Unknown profession")

    task_masked, pii_count = mask_pii(req.submitted_task)

    # Verification logic
    words = set(req.claimed_profession.lower().split())
    match = sum(1 for w in words if w in task_masked.lower())
    wc = len(task_masked.split())

    if not req.use_mock and is_available("gemini-2.0-flash"):
        try:
            from google import genai as g
            client = g.Client(api_key=os.getenv("GEMINI_API_KEY"))
            prompt = f"""Verify professional identity for Vantage AI.
Claimed profession: {req.claimed_profession}
Task (PII masked): {task_masked}
Does this task match a real {req.claimed_profession}'s work?
Respond ONLY with JSON: {{"verdict":"verified","confidence":0.88,"reasoning":"2 sentences","flags":[]}}
verdict: "verified" (>0.75), "review" (0.45-0.75), "flagged" (<0.45)"""
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, lambda: client.models.generate_content(
                model="gemini-2.0-flash", contents=prompt))
            text = resp.text.strip()
            if "```" in text: text = text.split("```")[1].lstrip("json").strip()
            result = json.loads(text)
        except:
            result = {"verdict":"verified" if match>=1 and wc>=15 else "review","confidence":0.75 if match>=1 else 0.55,"reasoning":"Verification completed.","flags":[]}
    else:
        if match >= 1 and wc >= 15:
            result = {"verdict":"verified","confidence":0.88,"reasoning":f"Task vocabulary aligns with {req.claimed_profession} work. Domain framing is consistent.","flags":[]}
        elif wc >= 10:
            result = {"verdict":"review","confidence":0.60,"reasoning":"Task is plausible but vocabulary is generic.","flags":["Task vocabulary could apply to multiple professions"]}
        else:
            result = {"verdict":"flagged","confidence":0.28,"reasoning":"Task too short or doesn't match profession.","flags":["Task too short"]}

    trust_score = user["trust_score"] if user else 0.5
    vote_weight = min(1.0, 0.5 + (user["vote_count"] / 40)) if user else 0.5

    return {**result,"trust_score":round(trust_score,2),"vote_weight":round(vote_weight,2),
            "dimensions":PROFESSIONS[req.claimed_profession]["dimensions"],
            "pii_masked":pii_count>0,"pii_count":pii_count,
            "is_authenticated":user is not None,
            "timestamp":datetime.utcnow().isoformat()}

@app.post("/taste-test")
async def taste_test(req: TasteTestRequest, user = Depends(get_current_user)):
    if len(req.task.strip()) < 20: raise HTTPException(400, "Task must be at least 20 characters")
    if req.profession not in PROFESSIONS: raise HTTPException(400, "Unknown profession")

    session_id = req.session_id or str(uuid.uuid4())
    task_masked, pii_count = mask_pii(req.task)
    system = get_system_prompt(req.profession)

    # RAG enrichment
    rag_contexts = []
    enriched_task = task_masked
    if rag and not req.use_mock:
        try:
            enriched_task, rag_contexts = await rag.enrich(task_masked, req.profession)
        except Exception as e:
            logger.debug("RAG enrichment failed: %s", e)

    # File context injection
    file_prefix = ""
    file_info = None
    if req.file_id and req.file_id in _file_cache:
        cached = _file_cache[req.file_id]
        file_prefix = build_file_context(cached["text"], cached["filename"], req.file_mode or "context")
        file_info = {"filename": cached["filename"], "mode": req.file_mode or "context", "chars": len(cached["text"])}

    # Build user message
    user_msg = file_prefix + USER_TEMPLATE.replace("{task}", enriched_task)

    models = DEFAULT_MODELS.copy()
    random.shuffle(models)
    labels = ["Model A", "Model B", "Model C"]

    async def one(model_id: str, label: str) -> dict:
        info = MODEL_REGISTRY.get(model_id, {})
        if req.use_mock or not is_available(model_id):
            response = get_mock(req.profession, model_id)
            latency_ms = random.uniform(800, 2200)
            source = "mock"
        else:
            response, latency_ms = await call_model(model_id, system, user_msg)
            source = "live"
        bias = check_bias(response)
        return {"label":label,"model":{**info,"id":model_id,"label":label,"source":source},
                "response":response,"latency_ms":round(latency_ms,1),"bias_flags":bias}

    responses = list(await asyncio.gather(*[one(models[i], labels[i]) for i in range(3)]))
    task_type = random.choice(PROFESSIONS[req.profession]["task_types"])
    all_bias = [f for r in responses for f in r.get("bias_flags",[])]

    with get_db() as conn:
        conn.execute("""INSERT OR REPLACE INTO sessions
            (id,user_id,profession,task,task_masked,task_type,responses,vote,pii_count,bias_flags,created_at)
            VALUES (?,?,?,?,?,?,?,NULL,?,?,?)""",
            (session_id, user["id"] if user else None, req.profession,
             req.task, task_masked, task_type, json.dumps(responses),
             pii_count, json.dumps(all_bias), datetime.utcnow().isoformat()))

    blind = [{"label":r["label"],"response":r["response"],"latency_ms":r["latency_ms"]} for r in responses]
    return {
        "session_id": session_id,
        "responses": blind,
        "task_type": task_type,
        "profession_dimensions": PROFESSIONS[req.profession]["dimensions"],
        "pii_masked": pii_count > 0,
        "pii_count": pii_count,
        "bias_flags": all_bias,
        "rag_used": len(rag_contexts) > 0,
        "rag_contexts": [{"profession":c["profession"],"topic":c["topic"]} for c in rag_contexts],
        "file_used": file_info is not None,
        "file_info": file_info,
        "model_status": [{"display":MODEL_REGISTRY[m]["display"],"provider":MODEL_REGISTRY[m]["provider"],
                          "live":is_available(m),"color":MODEL_REGISTRY[m]["color"]} for m in MODEL_REGISTRY],
    }

@app.post("/vote")
def vote(req: VoteRequest, user = Depends(get_current_user)):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (req.session_id,)).fetchone()
        if not row: raise HTTPException(404, "Session not found")
        if row["vote"] is not None: raise HTTPException(400, "Already voted")
        if req.chosen_index not in [0,1,2]: raise HTTPException(400, "chosen_index must be 0-2")

        responses = json.loads(row["responses"])
        winner = responses[req.chosen_index]
        winner_model = winner["model"]["display"]
        winner_color = winner["model"].get("color","#4285F4")
        winner_latency = winner.get("latency_ms",0)

        total = conn.execute("SELECT COUNT(*) FROM sessions WHERE profession=? AND vote IS NOT NULL",(row["profession"],)).fetchone()[0]+1
        same = conn.execute("SELECT COUNT(*) FROM sessions WHERE profession=? AND winner_model=? AND vote IS NOT NULL",(row["profession"],winner_model)).fetchone()[0]+1
        pci = compute_pci(winner_model, same, total)

        conn.execute("UPDATE sessions SET vote=?,winner_model=?,winner_latency_ms=?,voted_at=?,pci_score=? WHERE id=?",
                     (req.chosen_index,winner_model,winner_latency,datetime.utcnow().isoformat(),pci,req.session_id))

        # Update user trust score if authenticated
        if user:
            new_count = user["vote_count"] + 1
            new_trust = min(1.0, 0.5 + (new_count / 40))
            conn.execute("UPDATE users SET vote_count=?,trust_score=? WHERE id=?",
                         (new_count, new_trust, user["id"]))

    peer_pct = round((same/total)*100) if total > 2 else None
    reveal = [{"label":r["label"],"model_name":r["model"]["display"],"provider":r["model"]["provider"],
               "color":r["model"].get("color","#888"),"response":r["response"],
               "latency_ms":r.get("latency_ms",0),"won":i==req.chosen_index,
               "source":r["model"].get("source","mock")} for i,r in enumerate(responses)]

    return {
        "session_id":req.session_id,"chosen_index":req.chosen_index,
        "winner_label":winner["label"],"winner_model":winner_model,
        "winner_color":winner_color,"winner_latency_ms":winner_latency,"pci_score":pci,
        "surprise_message":random.choice([
            f"You preferred {winner['label']} — that was {winner_model}.",
            f"Your pick was {winner['label']} — {winner_model}.",
            f"{winner_model} won your vote as {winner['label']}.",
        ]),
        "peer_comparison":f"{peer_pct}% of {row['profession']}s preferred {winner_model}" if peer_pct else None,
        "reveal":reveal,"dimensions":PROFESSIONS[row["profession"]]["dimensions"],
        "user_vote_count":(user["vote_count"]+1) if user else None,
        "scorecard":{
            "profession":row["profession"],"task_type":row["task_type"],
            "task_preview":row["task"][:90]+"...","winner":winner_model,
            "winner_color":winner_color,"pci_score":pci,
            "share_text":f"I blind-tested GPT-4o vs Claude vs Gemini on a real {row['profession']} task — {winner_model} won. No brand bias. Try Vantage of AI.",
        }
    }

@app.post("/rate-dimensions")
def rate_dimensions(req: DimensionRatingRequest, user = Depends(get_current_user)):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (req.session_id,)).fetchone()
        if not row: raise HTTPException(404, "Session not found")
        if row["vote"] is None: raise HTTPException(400, "Vote first")
        for dim, score in req.ratings.items():
            if not (1<=score<=5): raise HTTPException(400, "Score must be 1-5")
            conn.execute("INSERT INTO dimension_ratings (session_id,user_id,profession,winner_model,dimension,score,created_at) VALUES (?,?,?,?,?,?,?)",
                (req.session_id, user["id"] if user else None, row["profession"],
                 row["winner_model"], dim, score, datetime.utcnow().isoformat()))
        conn.execute("UPDATE sessions SET dimension_scores=? WHERE id=?", (json.dumps(req.ratings),req.session_id))
    return {"status":"ok","ratings_saved":len(req.ratings)}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user = Depends(get_current_user)
):
    """
    Upload a file for context injection in taste test.
    Returns extracted text and a file_id to reference in taste-test.
    Max 10MB. Any file type supported.
    """
    MAX_SIZE = 10 * 1024 * 1024  # 10MB

    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(400, "File too large — maximum 10MB")

    filename = file.filename or "uploaded_file"
    mime_type = file.content_type or ""

    # Extract text
    text, method = extract_text(content, filename, mime_type)

    if not text.strip():
        raise HTTPException(400, "Could not extract text from this file. Try a different format.")

    # Store temporarily with a file_id (in-memory for now, Redis in production)
    file_id = uuid.uuid4().hex
    _file_cache[file_id] = {
        "filename": filename,
        "text": text,
        "method": method,
        "size": len(content),
        "uploaded_at": datetime.utcnow().isoformat(),
        "user_id": user["id"] if user else None,
    }

    return {
        "file_id": file_id,
        "filename": filename,
        "extraction_method": method,
        "chars_extracted": len(text),
        "preview": text[:200] + "..." if len(text) > 200 else text,
    }

# In-memory file cache (use Redis in production)
_file_cache: dict = {}

@app.get("/models-data")
def models_data():
    """
    Data endpoint for the public /models SEO page.
    Returns all models with PCI scores per profession.
    """
    with get_db() as conn:
        total_sessions = conn.execute("SELECT COUNT(*) FROM sessions WHERE vote IS NOT NULL").fetchone()[0]
        model_prof = conn.execute("""
            SELECT winner_model, profession, COUNT(*) as wins
            FROM sessions WHERE vote IS NOT NULL AND winner_model IS NOT NULL
            GROUP BY winner_model, profession
        """).fetchall()
        dim_scores = conn.execute("""
            SELECT winner_model, dimension, AVG(score) as avg, COUNT(*) as n
            FROM dimension_ratings GROUP BY winner_model, dimension
        """).fetchall()

    # Build per-model per-profession PCI
    results = {}
    for m_id, m_info in MODEL_REGISTRY.items():
        display = m_info["display"]
        results[display] = {
            "id": m_id,
            "display": display,
            "provider": m_info["provider"],
            "color": m_info["color"],
            "live": is_available(m_id),
            "context_window": m_info.get("context_window", 128000),
            "by_profession": {},
            "dimensions": {},
            "overall_pci": compute_pci(display, 0, max(total_sessions, 1)),
        }

    for row in model_prof:
        m = row["winner_model"]
        if m not in results: continue
        prof = row["profession"]
        pci = compute_pci(m, row["wins"], total_sessions)
        results[m]["by_profession"][prof] = {
            "wins": row["wins"],
            "pci": pci,
        }

    for row in dim_scores:
        m = row["winner_model"]
        if m not in results: continue
        results[m]["dimensions"][row["dimension"]] = {
            "avg": round(row["avg"], 2),
            "n": row["n"],
        }

    return {
        "models": list(results.values()),
        "total_votes": total_sessions,
        "professions": list(PROFESSIONS.keys()),
        "generated_at": datetime.utcnow().isoformat(),
    }

@app.get("/leaderboard")
def leaderboard(profession: Optional[str] = None):
    with get_db() as conn:
        if profession:
            rows = conn.execute("""SELECT winner_model,task_type,COUNT(*) as votes,AVG(winner_latency_ms) as avg_lat
                FROM sessions WHERE vote IS NOT NULL AND profession=?
                GROUP BY winner_model,task_type ORDER BY votes DESC""",(profession,)).fetchall()
        else:
            rows = conn.execute("""SELECT winner_model,profession,COUNT(*) as votes,AVG(winner_latency_ms) as avg_lat
                FROM sessions WHERE vote IS NOT NULL
                GROUP BY winner_model,profession ORDER BY votes DESC""").fetchall()

        dim_rows = conn.execute("""SELECT winner_model,dimension,AVG(score) as avg_score,COUNT(*) as n
            FROM dimension_ratings GROUP BY winner_model,dimension ORDER BY avg_score DESC""").fetchall()
        total_votes = conn.execute("SELECT COUNT(*) FROM sessions WHERE vote IS NOT NULL").fetchone()[0]
        total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

        # Trend data — votes per day per model (last 30 days)
        trend_rows = conn.execute("""SELECT winner_model, DATE(voted_at) as day, COUNT(*) as votes
            FROM sessions WHERE vote IS NOT NULL AND voted_at >= DATE('now','-30 days')
            GROUP BY winner_model, day ORDER BY day""").fetchall()

        prof_summary = conn.execute("""SELECT profession,winner_model,COUNT(*) as votes
            FROM sessions WHERE vote IS NOT NULL
            GROUP BY profession,winner_model ORDER BY profession,votes DESC""").fetchall()

    model_map = {v["display"]:{"id":k,**v} for k,v in MODEL_REGISTRY.items()}
    model_votes: dict = {}
    for r in rows:
        m = r["winner_model"]
        if m not in model_votes:
            info = model_map.get(m,{})
            wins = sum(rr["votes"] for rr in rows if rr["winner_model"]==m)
            pci = compute_pci(m, wins, total_sessions)
            model_votes[m] = {"model":m,"provider":info.get("provider",""),
                               "color":info.get("color","#888"),"total_votes":0,
                               "pci_score":pci,"avg_latency_ms":0,
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

    trend: dict = {}
    for r in trend_rows:
        m = r["winner_model"]
        if m not in trend: trend[m] = []
        trend[m].append({"day":r["day"],"votes":r["votes"]})

    return {"total_votes":total_votes,"total_sessions":total_sessions,"total_users":total_users,
            "rankings":rankings,"dimension_scores":dim_scores,
            "by_profession":prof_breakdown,"trend":trend,
            "generated_at":datetime.utcnow().isoformat()}

@app.get("/results")
def results():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM sessions WHERE vote IS NOT NULL ORDER BY voted_at DESC LIMIT 50").fetchall()
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        voted = conn.execute("SELECT COUNT(*) FROM sessions WHERE vote IS NOT NULL").fetchone()[0]
    return {"total_sessions":total,"completed":voted,"sessions":[dict(r) for r in rows]}

@app.get("/og/{session_id}")
def og_image(session_id: str):
    """
    Generate a 1200x630 OG image for a session scorecard.
    Used for LinkedIn/Twitter link previews.
    Install: pip install Pillow
    """
    from fastapi.responses import Response

    with get_db() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row or row["vote"] is None:
            raise HTTPException(404, "Session not found or not yet voted")

    try:
        from PIL import Image, ImageDraw, ImageFont
        import io

        # Canvas
        W, H = 1200, 630
        img = Image.new("RGB", (W, H), "#F5F3EF")
        draw = ImageDraw.Draw(img)

        # Model colors
        model_colors = {
            "Gemini 2.0 Flash": "#1B4FD8",
            "GPT-4o":           "#0D9373",
            "Claude Sonnet":    "#C2410C",
        }
        winner = row["winner_model"] or "AI Model"
        accent = model_colors.get(winner, "#1B4FD8")

        # Background accent bar (left side)
        draw.rectangle([0, 0, 8, H], fill=accent)

        # Top right accent blob
        draw.ellipse([900, -100, 1350, 350], fill=accent + "18")

        # Try to use a system font, fallback to default
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 72)
            font_med   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 36)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            font_mono  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18)
        except:
            font_large = ImageFont.load_default()
            font_med   = font_large
            font_small = font_large
            font_mono  = font_large

        # VANTAGE brand top-left
        draw.text((48, 48), "VANTAGE", font=font_small, fill="#9C9891")

        # Profession tag
        profession = row["profession"] or "Professional"
        draw.rectangle([48, 100, 48 + len(profession)*14 + 24, 136], fill=accent + "20", outline=accent + "40")
        draw.text((60, 108), profession.upper(), font=font_mono, fill=accent)

        # Winner headline
        draw.text((48, 160), "preferred", font=font_med, fill="#6B6760")
        draw.text((48, 210), winner, font=font_large, fill=accent)

        # Task preview
        task_preview = (row["task"] or "")[:80] + ("..." if len(row["task"] or "") > 80 else "")
        # Word wrap
        words = task_preview.split()
        lines, line = [], []
        for w in words:
            line.append(w)
            if len(' '.join(line)) > 52:
                lines.append(' '.join(line[:-1]))
                line = [w]
        if line: lines.append(' '.join(line))

        draw.line([48, 330, 8+48+4, 330], fill="#D9D6CF", width=1)
        y = 346
        for l in lines[:3]:
            draw.text((48, y), l, font=font_small, fill="#6B6760")
            y += 34

        # PCI score box
        pci = row["pci_score"]
        if pci:
            box_x, box_y = 48, 490
            draw.rectangle([box_x, box_y, box_x+200, box_y+60], fill=accent+"15", outline=accent+"30")
            draw.text((box_x+16, box_y+10), "PCI SCORE", font=font_mono, fill=accent)
            draw.text((box_x+16, box_y+32), f"{pci*100:.1f}", font=font_small, fill=accent)

        # Task type badge
        task_type = row["task_type"] or ""
        if task_type:
            draw.text((280, 506), task_type, font=font_mono, fill="#9C9891")

        # Bottom bar
        draw.rectangle([0, H-60, W, H], fill="#1A1814")
        draw.text((48, H-40), "vantageofai.com · Professional AI Benchmarking", font=font_mono, fill="#6B6760")
        draw.text((W-280, H-40), "No brand bias. Real professional votes.", font=font_mono, fill="#4B4845")

        # Right side — model comparison bars (if we have data)
        try:
            lb_data_raw = conn.execute("""SELECT winner_model, COUNT(*) as votes FROM sessions
                WHERE vote IS NOT NULL AND profession=? GROUP BY winner_model ORDER BY votes DESC""",
                (profession,)).fetchall() if False else []  # skip for now, use static
        except: pass

        # Simple right panel
        draw.rectangle([760, 80, 1160, 530], fill="white", outline="#E8E5DF")
        draw.text((800, 108), "Voted #1 by professionals", font=font_mono, fill="#9C9891")
        draw.text((800, 140), winner, font=font_med, fill=accent)

        # Three model indicators
        models = [("Gemini 2.0 Flash","#1B4FD8"), ("GPT-4o","#0D9373"), ("Claude Sonnet","#C2410C")]
        for i, (m, c) in enumerate(models):
            y_pos = 220 + i*80
            is_winner = m == winner
            bg = c+"15" if is_winner else "#F5F3EF"
            border = c if is_winner else "#E8E5DF"
            draw.rectangle([800, y_pos, 1140, y_pos+60], fill=bg, outline=border)
            draw.ellipse([816, y_pos+22, 832, y_pos+38], fill=c)
            draw.text((844, y_pos+18), m, font=font_small, fill="#1A1814" if is_winner else "#9C9891")
            if is_winner:
                draw.text((1060, y_pos+20), "WINNER", font=font_mono, fill=c)

        # Export
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return Response(content=buf.read(), media_type="image/png",
                       headers={"Cache-Control":"public, max-age=3600"})

    except ImportError:
        # Pillow not installed — return SVG fallback
        winner = row["winner_model"] or "AI Model"
        profession = row["profession"] or "Professional"
        svg = f'''<svg width="1200" height="630" xmlns="http://www.w3.org/2000/svg">
          <rect width="1200" height="630" fill="#F5F3EF"/>
          <rect width="8" height="630" fill="#1B4FD8"/>
          <text x="48" y="80" font-family="serif" font-size="20" fill="#9C9891">VANTAGE</text>
          <text x="48" y="200" font-family="serif" font-size="32" fill="#6B6760">preferred</text>
          <text x="48" y="280" font-family="serif" font-size="80" font-weight="bold" fill="#1B4FD8">{winner}</text>
          <text x="48" y="360" font-family="sans-serif" font-size="24" fill="#9C9891">{profession} · Blind AI Test</text>
          <rect y="570" width="1200" height="60" fill="#1A1814"/>
          <text x="48" y="607" font-family="monospace" font-size="16" fill="#6B6760">vantageofai.com · Professional AI Benchmarking</text>
        </svg>'''
        return Response(content=svg, media_type="image/svg+xml",
                       headers={"Cache-Control":"public, max-age=3600"})
