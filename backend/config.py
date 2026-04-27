import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")
VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", MODEL)

# Models the UI exposes for selection. First entry is the recommended default.
ALLOWED_MODELS: list[dict] = [
    {"id": "gpt-5",        "label": "GPT-5 (recommended)"},
    {"id": "gpt-5-mini",   "label": "GPT-5 mini — faster, cheaper"},
    {"id": "gpt-5-nano",   "label": "GPT-5 nano — cheapest"},
    {"id": "gpt-4.1",      "label": "GPT-4.1"},
    {"id": "gpt-4.1-mini", "label": "GPT-4.1 mini"},
    {"id": "gpt-4o",       "label": "GPT-4o"},
    {"id": "gpt-4o-mini",  "label": "GPT-4o mini"},
]
ALLOWED_MODEL_IDS = {m["id"] for m in ALLOWED_MODELS}

DATA = ROOT / "data"
PDF_DIR = DATA / "pdfs"
CACHE_DIR = DATA / "cache"
DB_PATH = DATA / "fiscal.sqlite"

for p in (DATA, PDF_DIR, CACHE_DIR):
    p.mkdir(parents=True, exist_ok=True)
