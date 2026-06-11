import tomllib
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    MODEL_NAME: str
    DEVICE: str
    DB_PATH: str
    DOCS_PATH: str

def load_config(config_path: str = "config.toml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        return AppConfig(
            MODEL_NAME="BAAI/bge-small-en-v1.5",
            DEVICE="cuda",
            DB_PATH="db/chroma_db",
            DOCS_PATH="docs"
        )
        
    with open(path, "rb") as f:
        data = tomllib.load(f)
        
    return AppConfig(
        MODEL_NAME=data["embedding"]["model_name"],
        DEVICE=data["embedding"]["device"],
        DB_PATH=data["paths"]["db_path"],
        DOCS_PATH=data["paths"]["docs_path"]
    )

config = load_config()