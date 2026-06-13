import tomllib
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    MODEL_NAME: str
    DEVICE: str

    LLM_MODEL_NAME: str
    LLM_BASE_URL: str
    LLM_NUM_CTX: int

    DB_PATH: str
    DOCS_PATH: str

def load_config(config_path: str = "config.toml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        return AppConfig(
            MODEL_NAME="BAAI/bge-small-en-v1.5",
            DEVICE="cuda",
            LLM_MODEL_NAME="qwen2.5:3b",
            LLM_BASE_URL="http://localhost:11434",
            LLM_NUM_CTX=4096,
            DB_PATH="db/chroma_db",
            DOCS_PATH="docs"
        )
        
    with open(path, "rb") as f:
        data = tomllib.load(f)
        
    return AppConfig(
        MODEL_NAME=data["embedding"]["model_name"],
        DEVICE=data["embedding"]["device"],
        LLM_MODEL_NAME=data["llm"]["model_name"],
        LLM_BASE_URL=data["llm"]["base_url"],
        LLM_NUM_CTX=data["llm"]["num_ctx"],
        DB_PATH=data["paths"]["db_path"],
        DOCS_PATH=data["paths"]["docs_path"]
    )

config = load_config()