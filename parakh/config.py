"""All settings in one place. Values come from .env (see .env.example)."""

import os

import dotenv

dotenv.load_dotenv()

# PostgreSQL
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = os.getenv("PGPORT", "5432")
DB_NAME = os.getenv("PGDATABASE")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

# Embedding models: model id -> name stored in the `model` column of
# ques_embed / ans_embed (also shown in the UI).
EMBEDDING_MODELS = {
    "krutrim-ai-labs/vyakyarth": "krutrim-ai-labs/vyakyarth",
    "BAAI/bge-m3": "bge-m3",
    "google/embeddinggemma-300m": "google/embeddinggemma-300m",
}
DEFAULT_EMBEDDING_MODEL = "krutrim-ai-labs/vyakyarth"

# LLM providers: "nvidia" (default), "openrouter" or "ollama".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia").lower()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_URL = os.getenv("NVIDIA_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
LLM_TIMEOUT_SECONDS = 90

# Photos are stored as relative paths ("Citizen/SurveyImages/x.jpg"); this
# server prefix turns them into viewable URLs. Unset -> no photos shown.
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "").rstrip("/")
