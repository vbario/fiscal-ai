import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")
VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", MODEL)

DATA = ROOT / "data"
PDF_DIR = DATA / "pdfs"
CACHE_DIR = DATA / "cache"
DB_PATH = DATA / "fiscal.sqlite"

for p in (DATA, PDF_DIR, CACHE_DIR):
    p.mkdir(parents=True, exist_ok=True)
