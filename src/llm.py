import ollama
import sys
import socket

from tqdm import tqdm
from langchain_ollama import ChatOllama
from src.config import AppConfig

def _ensure_ollama_server_is_running() -> None:
    """
    Checks if the Ollama server port is active. If closed, it prints
    a friendly troubleshooting instruction menu and halts the script.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2.0)
        if s.connect_ex(("localhost", 11434)) == 0:
            return

    print("\n" + "="*70)
    print("[Ollama Server Not Detected]")
    print("="*70)
    print("Your local AI execution engine is currently offline.")
    print("\nPlease verify the following steps to start the server:")
    print("  1. Ensure Ollama is installed on your machine.")
    print("  2. Open your terminal or command prompt and run:")
    print("     -> ollama serve")
    print("  3. Alternatively, launch the Ollama Desktop Application.")
    print("="*70)
    
    sys.exit(0)

def _ensure_model_exists(model_name: str) -> None:
    """
    Queries the local Ollama instance. If the model weights do not 
    exist on disk, it programmatically downloads them.
    """
    try:
        local_models = ollama.list()
        downloaded = any(model.model == model_name for model in local_models.models)
        
        if downloaded:
            return
            
        print(f"\n[Ollama] Model '{model_name}' not found locally.")

        consent = input("Do you want to download it now? (y/N): ").strip().lower()

        if consent.startswith("y"):
            print(f"Initiating download to your machine...")
            
            current_digest = ""
            pbar = None
            
            for progress in ollama.pull(model_name, stream=True):
                status = progress.get("status", "")
                digest = progress.get("digest", "")
                total = progress.get("total", 0)
                completed = progress.get("completed", 0)
                
                if digest:
                    if digest != current_digest:
                        if pbar:
                            pbar.close()
                        current_digest = digest
                        pbar = tqdm(
                            total=total, 
                            desc=f" Pulling {digest[7:19]}", 
                            unit="B", 
                            unit_scale=True
                        )
                    if pbar and completed:
                        pbar.n = completed
                        pbar.refresh()
                else:
                    if status and not status.startswith("pulling"):
                        print(f" -> {status}")
                        
            if pbar:
                pbar.close()
            print("[Ollama] Model setup finalized successfully!\n")
        else:
            print(f"\n[Error] Download cancelled. '{model_name}' is required to run retrieval.")
            print(f"Please run 'ollama pull {model_name}' manually or restart this script to consent.")
            sys.exit(0)
        
    except Exception as e:
        print(f"\n[Warning] Automated Ollama check failed: {e}")

def get_local_llm(app_config: AppConfig) -> ChatOllama:
    """Constructs a local ChatOllama instance"""
    
    _ensure_ollama_server_is_running()
    
    _ensure_model_exists(app_config.LLM_MODEL_NAME)
    
    return ChatOllama(
        base_url=app_config.LLM_BASE_URL,
        model=app_config.LLM_MODEL_NAME,
        num_ctx=app_config.LLM_NUM_CTX,
        temperature=0.2,
        keep_alive="24h"
    )