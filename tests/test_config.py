from pathlib import Path
from src.config import load_config, AppConfig

def test_load_config_structure():
    """Test that configuration initializes with expected attributes"""
    config = load_config(config_path="_non_existent_file_for_testing_.toml")
    
    assert isinstance(config, AppConfig)
    
    assert hasattr(config, "MODEL_NAME")
    assert isinstance(config.MODEL_NAME, str)
    
    assert hasattr(config, "DEVICE")
    assert isinstance(config.DEVICE, str)
    assert config.DEVICE.lower() in ("cuda", "cpu")

    assert hasattr(config, "LLM_MODEL_NAME")
    assert isinstance(config.LLM_MODEL_NAME, str)

    assert hasattr(config, "LLM_BASE_URL")
    assert isinstance(config.LLM_BASE_URL, str)
    
    assert hasattr(config, "DB_PATH")    
    assert isinstance(config.DB_PATH, str)
    assert Path(config.DB_PATH).name != "", "DB_PATH target directory resolution is empty"

    assert hasattr(config, "DOCS_PATH")    
    assert isinstance(config.DOCS_PATH, str)
    assert Path(config.DOCS_PATH).name != "", "DOCS_PATH target directory resolution is empty"

    assert hasattr(config, "LLM_NUM_CTX")
    assert isinstance(config.LLM_NUM_CTX, int)
    assert config.LLM_NUM_CTX > 0