import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = Path(os.getenv("TMPDIR", "/tmp")) / "suziee-ai" if os.getenv("VERCEL") else BASE_DIR
DATA_DIR = Path(os.getenv("SUZIEE_DATA_DIR", RUNTIME_DIR / "data"))
UPLOAD_DIR = Path(os.getenv("SUZIEE_UPLOAD_DIR", RUNTIME_DIR / "uploads"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").strip().lower()
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash") if AI_PROVIDER == "gemini" else os.getenv("CHAT_MODEL", "qwen2.5:3b")
VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", CHAT_MODEL) if AI_PROVIDER == "gemini" else os.getenv("VISION_MODEL", "llava:7b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
