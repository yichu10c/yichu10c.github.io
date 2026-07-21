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
    "You answer questions about Yihone — his work, skills, projects, experience, and background — "
    "using only the knowledge base provided below. Be concise but personable. "
    "If you don't know something, say you don't know rather than making it up.\n\n"
    f"## Knowledge Base\n\n{load_kb()}"
)


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
