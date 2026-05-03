"""
Vantage of AI — Backend v4
===========================
New in v4:
  - True multi-model: Gemini (live) + GPT-4o + Claude Sonnet
    → Each model falls back to high-quality mock if key not set
    → Add OPENAI_API_KEY / ANTHROPIC_API_KEY to .env to go fully live
  - Real LinkedIn scraping via Playwright (graceful fallback if blocked)
  - Improved verification using scraped profile data
  - All v3 features: SQLite, leaderboard, dimension scoring, peer comparison

.env setup:
  GEMINI_API_KEY=your_key        ← required (free at aistudio.google.com)
  OPENAI_API_KEY=your_key        ← optional (platform.openai.com)
  ANTHROPIC_API_KEY=your_key     ← optional (console.anthropic.com)

Run:
  pip install fastapi uvicorn google-generativeai openai anthropic python-dotenv playwright
  playwright install chromium
  uvicorn main:app --reload --port 8000
"""

import os, json, asyncio, uuid, random, sqlite3, re
from datetime import datetime
from typing import Optional
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Vantage of AI", version="4.0.0")
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
            task_type TEXT,
            responses TEXT,
            vote INTEGER,
            winner_model TEXT,
            dimension_scores TEXT,
            linkedin_profile TEXT,
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

# ── Models config ──────────────────────────────────────────────────────────

MODELS = [
    {
        "id":       "gemini-2.0-flash",
        "display":  "Gemini 2.0 Flash",
        "provider": "Google",
        "env_key":  "GEMINI_API_KEY",
        "color":    "#4285F4",
    },
    {
        "id":       "gpt-4o",
        "display":  "GPT-4o",
        "provider": "OpenAI",
        "env_key":  "OPENAI_API_KEY",
        "color":    "#10A37F",
    },
    {
        "id":       "claude-sonnet-4-20250514",
        "display":  "Claude Sonnet",
        "provider": "Anthropic",
        "env_key":  "ANTHROPIC_API_KEY",
        "color":    "#D97757",
    },
]

def model_available(model: dict) -> bool:
    return bool(os.getenv(model["env_key"]))

# ── Professions ────────────────────────────────────────────────────────────

PROFESSIONS = {
    "Data Scientist": {
        "dimensions": ["Statistical correctness", "Code quality", "Assumption clarity"],
        "task_types": ["Model selection", "EDA & visualization", "Feature engineering", "Evaluation & metrics", "Deployment & MLOps"],
        "prompt_persona": "You are a senior data scientist with 8 years of experience in ML, statistical modeling, and production ML systems.",
    },
    "Software Engineer": {
        "dimensions": ["Technical accuracy", "Code clarity", "Scalability thinking"],
        "task_types": ["System design", "Code review", "Debugging", "Architecture", "Performance"],
        "prompt_persona": "You are a senior software engineer with deep expertise in distributed systems, clean architecture, and production engineering.",
    },
    "ML Engineer": {
        "dimensions": ["Model correctness", "Production readiness", "Efficiency"],
        "task_types": ["Model optimization", "Pipeline design", "Serving & inference", "Monitoring", "Training infrastructure"],
        "prompt_persona": "You are a senior ML engineer who builds and ships production ML systems at scale.",
    },
    "Contract Attorney": {
        "dimensions": ["Legal accuracy", "Risk identification", "Practical advice"],
        "task_types": ["Contract review", "Risk assessment", "Negotiation strategy", "Compliance", "Dispute resolution"],
        "prompt_persona": "You are a senior contract attorney with 12 years of corporate law experience across SaaS, fintech, and enterprise deals.",
    },
    "Financial Analyst": {
        "dimensions": ["Analytical rigor", "Model accuracy", "Business insight"],
        "task_types": ["Valuation", "Financial modeling", "Due diligence", "Market analysis", "Risk modeling"],
        "prompt_persona": "You are a CFA-certified senior financial analyst at a top-tier investment bank.",
    },
    "Product Manager": {
        "dimensions": ["User focus", "Prioritization logic", "Measurability"],
        "task_types": ["PRD writing", "Prioritization", "Metrics & OKRs", "User research", "Roadmapping"],
        "prompt_persona": "You are a senior product manager with a track record of 0→1 products at Series B companies.",
    },
    "Radiologist": {
        "dimensions": ["Clinical accuracy", "Differential completeness", "Safety awareness"],
        "task_types": ["Image interpretation", "Differential diagnosis", "Report writing", "Protocol selection", "Findings correlation"],
        "prompt_persona": "You are a board-certified radiologist with subspecialty training in diagnostic imaging.",
    },
    "Marketing Manager": {
        "dimensions": ["Strategic clarity", "Audience targeting", "Measurable outcomes"],
        "task_types": ["Campaign strategy", "Copy & messaging", "Channel selection", "Analytics & attribution", "Brand positioning"],
        "prompt_persona": "You are a senior marketing manager who has led growth at multiple B2B SaaS companies.",
    },
    "UX Designer": {
        "dimensions": ["User empathy", "Design rationale", "Feasibility"],
        "task_types": ["User research", "Wireframing", "Usability review", "Design systems", "Accessibility"],
        "prompt_persona": "You are a senior UX designer with a portfolio spanning fintech, healthtech, and consumer apps.",
    },
    "Accountant / CPA": {
        "dimensions": ["Regulatory accuracy", "Practical application", "Risk flagging"],
        "task_types": ["Tax planning", "Financial reporting", "Audit prep", "Compliance", "Advisory"],
        "prompt_persona": "You are a CPA with 10 years in corporate accounting and tax advisory.",
    },
    "HR Manager": {
        "dimensions": ["Policy correctness", "Empathy & tone", "Actionability"],
        "task_types": ["Policy writing", "Performance management", "Conflict resolution", "Hiring & onboarding", "Compensation"],
        "prompt_persona": "You are an experienced HR manager who has built people ops from scratch at two high-growth startups.",
    },
    "Journalist": {
        "dimensions": ["Factual accuracy", "Source quality", "Narrative clarity"],
        "task_types": ["Story angle", "Research", "Headline writing", "Interview prep", "Fact-checking"],
        "prompt_persona": "You are an investigative journalist with 10 years at a national publication.",
    },
}

# ── Request / Response models ──────────────────────────────────────────────

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

# ── LinkedIn scraper ───────────────────────────────────────────────────────

async def scrape_linkedin(url: str) -> dict:
    """
    Scrapes a public LinkedIn profile page using Playwright.
    Returns extracted profile data. Falls back gracefully if blocked.
    """
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            ctx = await browser.new_context(user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ))
            page = await ctx.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            await asyncio.sleep(2)
            raw = await page.inner_text("body")
            await browser.close()

            lines = [l.strip() for l in raw.split("\n") if l.strip() and len(l.strip()) > 2]

            # Extract name (usually in first 15 lines, no special chars)
            name = None
            for line in lines[:15]:
                if len(line) > 3 and not any(c in line for c in ["·", "|", "@", "http", "Sign", "Log"]):
                    name = line
                    break

            # Extract headline
            headline = None
            if name and name in lines:
                idx = lines.index(name)
                if idx + 1 < len(lines):
                    headline = lines[idx + 1]

            # Extract current role + company
            current_role, company = None, None
            for line in lines:
                m = re.search(r"(.+?)\s+(?:at|@)\s+(.+)", line, re.IGNORECASE)
                if m and not current_role:
                    current_role = m.group(1).strip()
                    company = m.group(2).strip()

            return {
                "name": name,
                "headline": headline,
                "current_role": current_role or headline,
                "company": company,
                "raw_excerpt": raw[:2000],
                "scrape_success": True,
                "scrape_method": "playwright"
            }
    except ImportError:
        return {"scrape_success": False, "scrape_error": "Playwright not installed", "scrape_method": "none"}
    except Exception as e:
        return {"scrape_success": False, "scrape_error": str(e)[:200], "scrape_method": "none"}

def mock_linkedin_profile(profession: str) -> dict:
    return {
        "name": "Alex Chen",
        "headline": f"Senior {profession} · ex-Google · Open to opportunities",
        "current_role": f"Senior {profession}",
        "company": "TechCorp",
        "raw_excerpt": f"Senior {profession} with 7 years experience. Previously at Google, Stripe. Expert in the field.",
        "scrape_success": True,
        "scrape_method": "mock"
    }

# ── Gemini client ──────────────────────────────────────────────────────────

def get_gemini():
    import google.generativeai as genai
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(500, "GEMINI_API_KEY not set")
    genai.configure(api_key=key)
    return genai

# ── Verification ───────────────────────────────────────────────────────────

VERIFY_PROMPT = """You verify professional identities for Vantage of AI.

LinkedIn profile data:
{profile_text}

Claimed profession: {profession}
Submitted task: {task}

Evaluate:
1. Does the LinkedIn profile background match the claimed profession?
2. Does the task contain vocabulary and complexity consistent with a real {profession}?
3. Is there consistency between the profile and the task?

Respond ONLY with valid JSON, no markdown:
{{"verdict":"verified","confidence":0.88,"reasoning":"2 sentences","flags":[]}}

verdict: "verified" (>0.75), "review" (0.45–0.75), "flagged" (<0.45)"""

def mock_verify(profession: str, task: str) -> dict:
    words = set(profession.lower().split())
    match = sum(1 for w in words if w in task.lower())
    wc = len(task.split())
    if match >= 1 and wc >= 15:
        return {"verdict": "verified", "confidence": 0.88,
                "reasoning": f"Task vocabulary and complexity align with {profession} work. Domain framing is consistent.",
                "flags": []}
    elif wc >= 10:
        return {"verdict": "review", "confidence": 0.60,
                "reasoning": "Task is plausible but vocabulary is generic. Recommend human review.",
                "flags": ["Task vocabulary could apply to multiple professions"]}
    return {"verdict": "flagged", "confidence": 0.28,
            "reasoning": "Task is too short or doesn't match the claimed profession.",
            "flags": ["Task too short or mismatched"]}

async def gemini_verify(profession: str, task: str, profile: dict) -> dict:
    try:
        genai = get_gemini()
        loop = asyncio.get_event_loop()
        profile_text = (
            f"Name: {profile.get('name','Unknown')}\n"
            f"Headline: {profile.get('headline','N/A')}\n"
            f"Role: {profile.get('current_role','N/A')}\n"
            f"Company: {profile.get('company','N/A')}\n"
            f"Excerpt: {profile.get('raw_excerpt','N/A')[:500]}"
        )
        prompt = VERIFY_PROMPT.format(
            profile_text=profile_text, profession=profession, task=task)
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = await loop.run_in_executor(None, lambda: model.generate_content(prompt))
        text = resp.text.strip()
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return mock_verify(profession, task)
    except Exception as e:
        raise HTTPException(500, f"Gemini verify error: {e}")

# ── Taste test — 3 real models ─────────────────────────────────────────────

TASTE_PROMPT = """{persona}

Your colleague asked for expert help on the following professional task.
Be specific, actionable, and grounded in domain expertise. Under 200 words.
Do not say which AI you are.

Task: {task}"""

# High-quality mock responses per profession per "model personality"
# Each model has a distinct style: Flash=direct/tactical, GPT=structured, Claude=nuanced
MOCK_STYLES = {
    "gemini-2.0-flash": "Direct and tactical. Lead with the answer, then explain.",
    "gpt-4o":           "Structured with clear sections. Thorough and comprehensive.",
    "claude-sonnet-4-20250514": "Nuanced and thoughtful. Acknowledge tradeoffs before recommending.",
}

MOCK_RESPONSES = {
    "Data Scientist": {
        "gemini-2.0-flash": "Switch to XGBoost with scale_pos_weight=11.5 first — fastest path to better recall. Then lower your classification threshold to 0.2–0.3 using the precision-recall curve, not ROC (ROC is misleading at 8% imbalance). Only add SMOTE if recall stays below 65% after both changes. Your features matter more than your model — check SHAP values to confirm churn signals (usage drop, support tickets) have high importance before assuming the model is the problem.",
        "gpt-4o": "Three things to address in order:\n\n1. Threshold tuning: Your 0.5 threshold optimises accuracy, not recall. Use precision-recall curve to find the optimal threshold — expect recall to jump to 65–70% with minimal precision loss.\n\n2. Model upgrade: XGBoost with class_weight='balanced' will outperform logistic regression here. Set scale_pos_weight to ~11.5 (ratio of negatives to positives).\n\n3. SMOTE: Avoid unless you're below 1% churn rate. It creates synthetic samples that don't reflect real churn patterns and can hurt generalisation. Use stratified k-fold CV with F1 as your metric throughout.",
        "claude-sonnet-4-20250514": "Before changing models, I'd diagnose why recall is low. Run SHAP values on your logistic regression — if churn features (session frequency, plan downgrades, support tickets) have low importance, the problem is feature engineering, not the model. XGBoost won't fix weak features.\n\nIf features look healthy, your fastest win is threshold adjustment — move from 0.5 to 0.25 and recheck recall. Then try XGBoost with scale_pos_weight=11.5. SMOTE is worth trying only as a last resort; synthetic samples often don't capture real churn behaviour.\n\nOne more thing: at 8% imbalance, accuracy is a useless metric. Switch your evaluation to F1-score or AUC-PR from the start.",
    },
    "Software Engineer": {
        "gemini-2.0-flash": "Write-through caching with Redlock (3 Redis nodes) and 300s TTL + ±30s jitter. That's it. The jitter eliminates thundering herd; Redlock prevents the race condition. Add a circuit breaker so Redis failures fall through to Postgres cleanly. ~2ms latency cost, stampede risk eliminated.",
        "gpt-4o": "Three options, in order of complexity:\n\n1. Write-through + jitter: Set TTL to 300s ±30s random jitter. Simple, eliminates thundering herd for most cases.\n\n2. Cache-aside with optimistic locking: Read cache → miss → read DB → compare-and-set with version tag. Stronger consistency without distributed locking.\n\n3. Redlock (if #2 isn't enough): Three Redis nodes minimum for correctness. Adds ~2ms latency but gives you distributed mutual exclusion.\n\nAlso: 60s TTL is too aggressive at your traffic level — you're making ~1.6M unnecessary DB calls/day. Bump to 600s and invalidate explicitly on writes.",
        "claude-sonnet-4-20250514": "The stampede risk and the race condition are two separate problems — worth addressing individually.\n\nFor stampede: TTL jitter (±30s on a 300s TTL) solves 90% of cases without distributed locking complexity. Add probabilistic early expiration (PER algorithm) for the remaining cases.\n\nFor the race condition: cache-aside with optimistic locking is the right pattern — compare-and-set with a version tag before writing back to cache. Discard and re-read if the version changed.\n\nRedlock is the nuclear option — correct, but it adds operational complexity (three Redis nodes, failure modes). I'd reach for it only if the above don't work.\n\nSide note: your 60s TTL is generating ~1.6M unnecessary DB hits/day per key. Bump to 600s and invalidate on writes — immediate win.",
    },
    "Contract Attorney": {
        "gemini-2.0-flash": "§8.2 indemnification is overbroad — 'any and all claims arising out of or relating to' covers the indemnitee's own negligence under Delaware law. Narrow to: 'claims arising out of the indemnifying party's material breach, gross negligence, or wilful misconduct.' Add a contributory negligence carve-out. Standard SaaS language, shouldn't be controversial.",
        "gpt-4o": "Three issues to address:\n\n1. Indemnification scope (§8.2): Current language is overbroad — covers indemnitee's own negligence. Narrow to material breach, gross negligence, wilful misconduct. Add carve-out for claims the indemnitee contributed to.\n\n2. Liability cap structure: Uncapped carve-outs for IP and confidentiality are asymmetric for a smaller vendor. Negotiate a separate cap (3x annual fees) rather than fully uncapped.\n\n3. Force majeure: Post-COVID, 'acts of God, war, terrorism' is insufficient. Add explicit language for pandemics, government-mandated closures, and supply chain disruptions. Include a 5-business-day notice requirement and mitigation obligation.",
        "claude-sonnet-4-20250514": "I'd flag three issues, in order of urgency:\n\nThe indemnification clause in §8.2 is the most urgent — 'arising out of or relating to' is language Delaware courts read broadly, often including the indemnitee's own negligence. This is a real risk, not a theoretical one. Narrow the trigger to material breach, gross negligence, or wilful misconduct.\n\nThe limitation of liability structure has asymmetric risk: uncapped carve-outs for IP infringement and confidentiality on both sides effectively neutralise the cap for the situations that matter most. A separate, higher cap (3x annual fees) is a reasonable middle ground.\n\nFinally, the force majeure clause needs updating — courts have split on whether COVID-era disruptions qualify under standard boilerplate. Explicit language covering pandemics and government closures protects both parties.",
    },
}

def get_mock_response(profession: str, model_id: str, task: str) -> str:
    prof_data = MOCK_RESPONSES.get(profession, MOCK_RESPONSES.get("Software Engineer", {}))
    resp = prof_data.get(model_id)
    if resp:
        return resp
    # Generic fallback with model style
    style = MOCK_STYLES.get(model_id, "Be specific and actionable.")
    return f"[Mock response for {profession} task — style: {style}]\n\nThis would be a real {profession} expert answer from this model when API key is configured."

async def call_gemini(task: str, profession: str) -> str:
    genai = get_gemini()
    loop = asyncio.get_event_loop()
    persona = PROFESSIONS[profession]["prompt_persona"]
    prompt = TASTE_PROMPT.format(persona=persona, task=task)
    model = genai.GenerativeModel("gemini-2.0-flash")
    resp = await loop.run_in_executor(None, lambda: model.generate_content(prompt))
    return resp.text.strip()

async def call_openai(task: str, profession: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    persona = PROFESSIONS[profession]["prompt_persona"]
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, lambda: client.chat.completions.create(
        model="gpt-4o",
        max_tokens=400,
        messages=[
            {"role": "system", "content": persona + " Do not reveal which AI you are."},
            {"role": "user", "content": f"Professional task:\n{task}"}
        ]
    ))
    return resp.choices[0].message.content.strip()

async def call_anthropic(task: str, profession: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    persona = PROFESSIONS[profession]["prompt_persona"]
    loop = asyncio.get_event_loop()
    msg = await loop.run_in_executor(None, lambda: client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        system=persona + " Do not reveal which AI you are. Keep response under 200 words.",
        messages=[{"role": "user", "content": f"Professional task:\n{task}"}]
    ))
    return msg.content[0].text.strip()

async def get_model_response(model: dict, task: str, profession: str, use_mock: bool) -> dict:
    label = model["display"]
    try:
        if use_mock or not model_available(model):
            response = get_mock_response(profession, model["id"], task)
            source = "mock"
        elif model["id"] == "gemini-2.0-flash":
            response = await call_gemini(task, profession)
            source = "live"
        elif model["id"] == "gpt-4o":
            response = await call_openai(task, profession)
            source = "live"
        elif model["id"].startswith("claude"):
            response = await call_anthropic(task, profession)
            source = "live"
        else:
            response = get_mock_response(profession, model["id"], task)
            source = "mock"
    except Exception as e:
        response = get_mock_response(profession, model["id"], task)
        source = f"mock (error: {str(e)[:60]})"

    return {
        "model": {**model, "source": source},
        "response": response,
        "display": label,
    }

async def run_taste_test(task: str, profession: str, use_mock: bool) -> list:
    models = MODELS.copy()
    random.shuffle(models)
    labels = ["Model A", "Model B", "Model C"]

    results = await asyncio.gather(*[
        get_model_response(models[i], task, profession, use_mock)
        for i in range(3)
    ])

    return [
        {**r, "label": labels[i], "model": {**r["model"], "label": labels[i]}}
        for i, r in enumerate(results)
    ]

# ── Task classification ────────────────────────────────────────────────────

async def classify_task(task: str, profession: str) -> str:
    try:
        types = PROFESSIONS[profession]["task_types"]
        genai = get_gemini()
        loop = asyncio.get_event_loop()
        prompt = f"Classify this {profession} task into exactly one: {', '.join(types)}\n\nTask: {task}\n\nRespond with ONLY the category name."
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = await loop.run_in_executor(None, lambda: model.generate_content(prompt))
        result = resp.text.strip()
        return result if result in types else types[0]
    except:
        return PROFESSIONS.get(profession, {}).get("task_types", ["General"])[0]

# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Vantage of AI",
        "version": "4.0.0",
        "models": [{"id": m["id"], "display": m["display"], "live": model_available(m)} for m in MODELS],
        "professions": list(PROFESSIONS.keys()),
    }

@app.get("/health")
def health():
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        voted = conn.execute("SELECT COUNT(*) FROM sessions WHERE vote IS NOT NULL").fetchone()[0]
    return {
        "status": "ok",
        "models_live": {m["display"]: model_available(m) for m in MODELS},
        "total_sessions": total,
        "completed_votes": voted,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.get("/professions")
def get_professions():
    return {name: {"dimensions": cfg["dimensions"], "task_types": cfg["task_types"]}
            for name, cfg in PROFESSIONS.items()}

@app.get("/models")
def get_models():
    return [{"display": m["display"], "provider": m["provider"],
             "live": model_available(m), "color": m["color"]} for m in MODELS]

@app.post("/verify")
async def verify(req: VerifyRequest):
    if not req.linkedin_url.startswith("https://www.linkedin.com/in/"):
        raise HTTPException(400, "URL must start with https://www.linkedin.com/in/")
    if len(req.submitted_task.strip()) < 20:
        raise HTTPException(400, "Task must be at least 20 characters")
    if req.claimed_profession not in PROFESSIONS:
        raise HTTPException(400, f"Unknown profession")

    # Scrape LinkedIn
    if req.use_mock:
        profile = mock_linkedin_profile(req.claimed_profession)
    else:
        profile = await scrape_linkedin(req.linkedin_url)

    # Verify
    if req.use_mock:
        result = mock_verify(req.claimed_profession, req.submitted_task)
    else:
        result = await gemini_verify(req.claimed_profession, req.submitted_task, profile)

    return {
        **result,
        "trust_score": round(0.5 + result["confidence"] * 0.2, 2),
        "vote_weight": 0.5,
        "dimensions": PROFESSIONS[req.claimed_profession]["dimensions"],
        "profile": profile,
        "timestamp": datetime.utcnow().isoformat(),
    }

@app.post("/taste-test")
async def taste_test(req: TasteTestRequest):
    if len(req.task.strip()) < 20:
        raise HTTPException(400, "Task must be at least 20 characters")
    if req.profession not in PROFESSIONS:
        raise HTTPException(400, "Unknown profession")

    session_id = req.session_id or str(uuid.uuid4())

    # Classify task
    task_type = random.choice(PROFESSIONS[req.profession]["task_types"]) if req.use_mock \
                else await classify_task(req.task, req.profession)

    # Get model responses
    responses = await run_taste_test(req.task, req.profession, req.use_mock)

    blind = [{"label": r["label"], "response": r["response"]} for r in responses]

    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO sessions
            (id, profession, task, task_type, responses, vote, created_at)
            VALUES (?, ?, ?, ?, ?, NULL, ?)
        """, (session_id, req.profession, req.task, task_type,
              json.dumps(responses), datetime.utcnow().isoformat()))

    # Tell frontend which models are live
    model_status = [
        {"display": m["display"], "provider": m["provider"],
         "live": model_available(m), "color": m["color"]}
        for m in MODELS
    ]

    return {
        "session_id": session_id,
        "responses": blind,
        "task_type": task_type,
        "profession_dimensions": PROFESSIONS[req.profession]["dimensions"],
        "model_status": model_status,
    }

@app.post("/vote")
def vote(req: VoteRequest):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (req.session_id,)).fetchone()
        if not row: raise HTTPException(404, "Session not found")
        if row["vote"] is not None: raise HTTPException(400, "Already voted")
        if req.chosen_index not in [0, 1, 2]: raise HTTPException(400, "chosen_index must be 0–2")

        responses = json.loads(row["responses"])
        winner = responses[req.chosen_index]
        winner_model = winner["model"]["display"]
        winner_color = winner["model"].get("color", "#4285F4")

        conn.execute("UPDATE sessions SET vote=?, winner_model=?, voted_at=? WHERE id=?",
                     (req.chosen_index, winner_model, datetime.utcnow().isoformat(), req.session_id))

        total = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE profession=? AND vote IS NOT NULL",
            (row["profession"],)).fetchone()[0] + 1
        same = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE profession=? AND winner_model=? AND vote IS NOT NULL",
            (row["profession"], winner_model)).fetchone()[0] + 1

    peer_pct = round((same / total) * 100) if total > 2 else None

    reveal = [
        {"label": r["label"], "model_name": r["model"]["display"],
         "provider": r["model"]["provider"], "color": r["model"].get("color","#888"),
         "response": r["response"], "won": i == req.chosen_index,
         "source": r["model"].get("source","mock")}
        for i, r in enumerate(responses)
    ]

    return {
        "session_id": req.session_id,
        "chosen_index": req.chosen_index,
        "winner_label": winner["label"],
        "winner_model": winner_model,
        "winner_color": winner_color,
        "surprise_message": random.choice([
            f"You preferred {winner['label']} — that was {winner_model}.",
            f"Your pick was {winner['label']} — {winner_model}.",
            f"{winner_model} won your vote as {winner['label']}.",
        ]),
        "peer_comparison": f"{peer_pct}% of {row['profession']}s preferred {winner_model}" if peer_pct else None,
        "reveal": reveal,
        "dimensions": PROFESSIONS[row["profession"]]["dimensions"],
        "scorecard": {
            "profession": row["profession"],
            "task_type": row["task_type"],
            "task_preview": row["task"][:90] + "...",
            "winner": winner_model,
            "winner_color": winner_color,
            "share_text": f"I blind-tested GPT-4o vs Claude vs Gemini on a real {row['profession']} task — {winner_model} won. No brand bias. Try it: vantageofai.com",
        }
    }

@app.post("/rate-dimensions")
def rate_dimensions(req: DimensionRatingRequest):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (req.session_id,)).fetchone()
        if not row: raise HTTPException(404, "Session not found")
        if row["vote"] is None: raise HTTPException(400, "Vote first")
        for dim, score in req.ratings.items():
            if not (1 <= score <= 5): raise HTTPException(400, f"Score must be 1–5")
            conn.execute("""INSERT INTO dimension_ratings
                (session_id, profession, winner_model, dimension, score, created_at)
                VALUES (?,?,?,?,?,?)""",
                (req.session_id, row["profession"], row["winner_model"],
                 dim, score, datetime.utcnow().isoformat()))
        conn.execute("UPDATE sessions SET dimension_scores=? WHERE id=?",
                     (json.dumps(req.ratings), req.session_id))
    return {"status": "ok", "ratings_saved": len(req.ratings)}

@app.get("/leaderboard")
def leaderboard(profession: Optional[str] = None):
    with get_db() as conn:
        if profession:
            rows = conn.execute("""
                SELECT winner_model, task_type, COUNT(*) as votes
                FROM sessions WHERE vote IS NOT NULL AND profession=?
                GROUP BY winner_model, task_type ORDER BY votes DESC
            """, (profession,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT winner_model, profession, COUNT(*) as votes
                FROM sessions WHERE vote IS NOT NULL
                GROUP BY winner_model, profession ORDER BY votes DESC
            """).fetchall()

        dim_rows = conn.execute("""
            SELECT winner_model, dimension, AVG(score) as avg_score, COUNT(*) as n
            FROM dimension_ratings GROUP BY winner_model, dimension ORDER BY avg_score DESC
        """).fetchall()

        total_votes = conn.execute("SELECT COUNT(*) FROM sessions WHERE vote IS NOT NULL").fetchone()[0]
        total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

        prof_summary = conn.execute("""
            SELECT profession, winner_model, COUNT(*) as votes
            FROM sessions WHERE vote IS NOT NULL
            GROUP BY profession, winner_model ORDER BY profession, votes DESC
        """).fetchall()

    model_map = {m["display"]: m for m in MODELS}
    model_votes: dict = {}
    for r in rows:
        m = r["winner_model"]
        if m not in model_votes:
            info = model_map.get(m, {})
            model_votes[m] = {"model": m, "provider": info.get("provider",""),
                               "color": info.get("color","#888"),
                               "total_votes": 0, "by_profession": {}, "by_task_type": {}}
        model_votes[m]["total_votes"] += r["votes"]
        key = "by_task_type" if profession else "by_profession"
        sub = r["task_type"] if profession else r["profession"]
        model_votes[m][key][sub] = r["votes"]

    rankings = sorted(model_votes.values(), key=lambda x: x["total_votes"], reverse=True)
    for i, r in enumerate(rankings): r["rank"] = i + 1

    dim_scores: dict = {}
    for r in dim_rows:
        m = r["winner_model"]
        if m not in dim_scores: dim_scores[m] = {}
        dim_scores[m][r["dimension"]] = {"avg": round(r["avg_score"], 2), "n": r["n"]}

    prof_breakdown: dict = {}
    for r in prof_summary:
        if r["profession"] not in prof_breakdown: prof_breakdown[r["profession"]] = []
        prof_breakdown[r["profession"]].append({"model": r["winner_model"], "votes": r["votes"]})

    return {"total_votes": total_votes, "total_sessions": total_sessions,
            "rankings": rankings, "dimension_scores": dim_scores,
            "by_profession": prof_breakdown, "generated_at": datetime.utcnow().isoformat()}

@app.get("/results")
def results():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE vote IS NOT NULL ORDER BY voted_at DESC LIMIT 50"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        voted = conn.execute("SELECT COUNT(*) FROM sessions WHERE vote IS NOT NULL").fetchone()[0]
    return {"total_sessions": total, "completed": voted, "sessions": [dict(r) for r in rows]}
