"""
FastAPI server for Yihone's portfolio chat.
Calls MiniMax API to answer questions about Yihone using the KB.
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx


# ── Config ──────────────────────────────────────────────────────────────────
KB_DIR = Path(__file__).parent          # /home/ubuntu/yichu10c.github.io/kb/
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_MODEL = "MiniMax-Text-01"

app = FastAPI(title="Yihone Portfolio Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── KB Loading ──────────────────────────────────────────────────────────────
def load_kb() -> str:
    """Concatenate all .md files in the kb/ folder into one context string."""
    parts = []
    for md_file in sorted(KB_DIR.glob("*.md")):
        parts.append(f"## {md_file.stem.replace('_', ' ').title()}\n\n{md_file.read_text()}")
    return "\n\n---\n\n".join(parts)


SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable AI assistant representing Yihone Chu. "
    "You ONLY answer questions about Yihone — his work, skills, projects, experience, education, and background. "
    "You also help visitors get in touch with him. "
    "If a question is unrelated to Yihone — math problems, general knowledge, current events, or anything else — "
    "politely decline and offer to help with something about Yihone instead.\n\n"

    # ── Response guidelines ────────────────────────────────────────────────
    "HOW TO TALK ABOUT HIS WORK:\n\n"
    "Write like a real person describing a colleague — not like a resume or a factsheet. "
    "Use complete sentences, vary your sentence structure, and naturally integrate details "
    "rather than listing them as separate facts. When describing his work, imagine you are "
    "verbally explaining what he does to someone at a meetup or on a phone call.\n\n"

    "EXAMPLE — How to talk about his Nationwide role:\n"
    "\"Yihone works as a Software Engineer at Nationwide Insurance, primarily on the backend side "
    "of their financial products. He's been deeply involved in triaging and fixing production "
    "issues — digging into root causes, coordinating hotfixes, and keeping things stable. On the "
    "infrastructure side, he builds high-throughput Java APIs and manages Oracle SQL schema work "
    "to keep query performance solid. He also plays a role in release coordination across "
    "their large engineering org, and has been building out automated cron pipelines to handle "
    "repetitive operational work.\"\n\n"

    "STRUCTURE:\n"
    "• 2–4 sentences per topic, written as flowing prose\n"
    "• No bullet points, no dashes, no numbered lists, no asterisks\n"
    "• Use line breaks between topics (one blank line)\n"
    "• Keep it conversational — like a person explaining, not a data sheet\n\n"

    "TONE: Natural, warm, professional. Like a coworker giving a 30-second intro.\n\n"

    # ── KB ─────────────────────────────────────────────────────────────────
    f"## Knowledge Base\n\n{load_kb()}"
)


def is_related_to_yihone(question: str) -> bool:
    """Block clearly off-topic questions; let everything else through to the model."""
    q = question.lower().strip()

    # Hard block: math, calculators, weather, definitions, unrelated general knowledge
    off_topic = [
        "1+1", "2+2", "3+3", "what's ", "calculate", "math",
        "weather", "news", "bitcoin", "stock price", "sports score",
        "recipe", "translate ", "convert ", "capital of", "who is ",
        "what year", "what country", "define ", "meaning of",
    ]
    if any(p in q for p in off_topic):
        return False

    # Allow if it mentions Yihone or portfolio-adjacent topics
    allow = ["yihone", "yichu", "your portfolio", "your website",
             "about you", "you built", "you work", "your projects",
             "your skills", "your experience", "contact you",
             "reach you", "get in touch", "your background"]
    if any(p in q for p in allow):
        return True

    # Default: let the model decide (generous — better UX than blocking valid questions)
    return True


DECLINE_RESPONSES = [
    "I'm here to answer questions about Yihone! Try asking about his projects, skills, or experience instead.",
    "This chat is focused on Yihone — feel free to ask about his work, background, or how to reach him!",
    "I'm only able to help with questions about Yihone. Want to know about his projects or experience?",
]


# ── Request / Response models ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


# ── Routes ───────────────────────────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not MINIMAX_API_KEY:
        raise HTTPException(status_code=500, detail="MiniMax API key not configured")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not is_related_to_yihone(req.question):
        import random
        return ChatResponse(answer=random.choice(DECLINE_RESPONSES))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": req.question},
    ]

    payload = {
        "model": MINIMAX_MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{MINIMAX_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    choice = data["choices"][0]
    answer = choice["message"]["content"]

    return ChatResponse(answer=answer)


@app.get("/api/health")
async def health():
    return {"status": "ok", "kb_files": len(list(KB_DIR.glob("*.md")))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
