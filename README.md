# Vantage of AI — v2

Full blind taste test + professional verification. Run locally in 5 minutes.

## What's built

| Screen | What it does |
|--------|-------------|
| **Home** | Landing page explaining the product |
| **Verify** | LinkedIn URL + task → Claude checks legitimacy |
| **Blind Test** | Submit task → 3 anonymous model responses |
| **Vote** | Pick the best response |
| **Reveal** | See which model won + shareable card |

---

## Run it (5 minutes)

### 1. Install backend
```bash
cd backend
pip install -r requirements.txt
playwright install chromium
```

### 2. Start backend
```bash
uvicorn main:app --reload --port 8000
```
You should see: `Uvicorn running on http://127.0.0.1:8000`

### 3. Open frontend
```bash
open frontend/index.html
```
Or just double-click `frontend/index.html` in Finder / Explorer.

### 4. Test it
- Keep **Mock Mode ON** (toggle in top-right nav)
- Click "Start the test"
- Select a profession, paste any LinkedIn URL, paste a task
- Go through verify → blind test → vote → reveal

---

## Add real API keys (when ready)

```bash
cp backend/.env.example backend/.env
# Edit .env:
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

Then toggle **Mock Mode OFF** in the UI (top-right nav button turns green "LIVE MODE").

When live:
- Verification calls real Claude for profile + task plausibility check
- Taste test calls Claude 3 times with different system prompts to simulate 3 models
- To add real GPT-4 and Gemini: see `real_model_responses()` in `backend/main.py`

---

## API endpoints

```
POST /verify        Verify LinkedIn URL + task
POST /taste-test    Get 3 anonymous model responses for a task
POST /vote          Submit vote → get reveal + scorecard
GET  /results       See all completed sessions (your data)
GET  /health        Health check
```

### Example: run a taste test
```bash
curl -X POST http://localhost:8000/taste-test \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Review this distributed caching architecture: Redis write-through with 60s TTL...",
    "profession": "Software Engineer",
    "use_mock": true
  }'
```

### See your data accumulating
```bash
curl http://localhost:8000/results
```
This is how you prove to YC that real professionals are using it.

---

## Project structure

```
vantage/
├── backend/
│   ├── main.py              All API logic — verify, taste test, vote, reveal
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── index.html           Entire UI — home, verify, test, vote, reveal, share card
```

---

## What to build next

1. **SQLite persistence** — right now sessions are in-memory and reset on restart. Add SQLite so votes survive restarts and you can show real accumulated data.
2. **Real multi-model** — swap `real_model_responses()` in `main.py` to call GPT-4o and Gemini in parallel alongside Claude.
3. **LinkedIn scraping** — the Playwright scraper is in verification v1 (`verify` endpoint). Plug it in to replace mock profile data.
4. **Session IDs in URL** — so users can share their reveal screen directly.
5. **Results dashboard** — a `/leaderboard` page showing aggregate votes across all sessions.
