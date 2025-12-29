"""
Path constants for Open-LLM-VTuber.

This module defines all path constants used throughout the application.
Paths can be overridden using environment variables for flexibility.
"""

import os
from pathlib import Path

# Base directory (project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configuration paths
CONFIG_DIR = Path(os.getenv("OLV_CONFIG_DIR", PROJECT_ROOT / "config"))
MODEL_DICT_PATH = Path(os.getenv("OLV_MODEL_DICT_PATH", CONFIG_DIR / "model_dict.json"))
MCP_SERVERS_PATH = Path(os.getenv("OLV_MCP_SERVERS_PATH", CONFIG_DIR / "mcp_servers.json"))

# Model storage paths
MODELS_DIR = Path(os.getenv("OLV_MODELS_DIR", PROJECT_ROOT / "models"))
PIPER_MODELS_DIR = Path(os.getenv("OLV_PIPER_MODELS_DIR", MODELS_DIR / "piper"))
SHERPA_ONNX_MODELS_DIR = Path(os.getenv("OLV_SHERPA_ONNX_MODELS_DIR", MODELS_DIR))

# Default model paths
DEFAULT_PIPER_MODEL = Path(
    os.getenv("OLV_PIPER_MODEL_PATH", PIPER_MODELS_DIR / "ko_KR-medium.onnx")
)
DEFAULT_SHERPA_ONNX_SENSE_VOICE = Path(
    os.getenv(
        "OLV_SHERPA_ONNX_SENSE_VOICE_PATH",
        SHERPA_ONNX_MODELS_DIR / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
    )
)

# Ensure directories exist
def ensure_directories():
    """Ensure all required directories exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PIPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SHERPA_ONNX_MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Initialize directories on import
ensure_directories()

