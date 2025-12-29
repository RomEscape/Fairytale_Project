"""
Utility functions for downloading models from HuggingFace.

This module provides functions to automatically download models from HuggingFace
when they are not found locally, similar to how sherpa-onnx models are handled.
"""

import os
import subprocess
from pathlib import Path
from loguru import logger


def download_huggingface_model(
    repo_id: str,
    filename: str,
    local_dir: str | Path,
    subfolder: str | None = None,
) -> Path:
    """
    Download a model file from HuggingFace using huggingface-cli.

    Args:
        repo_id: HuggingFace repository ID (e.g., "PJiNH/snow_white_gguf")
        filename: Name of the file to download (e.g., "model-q4_0.gguf")
        local_dir: Local directory to save the file
        subfolder: Optional subfolder in the repository

    Returns:
        Path to the downloaded file

    Raises:
        FileNotFoundError: If huggingface-cli is not installed
        subprocess.CalledProcessError: If download fails
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Check if file already exists
    file_path = local_dir / filename
    if file_path.exists():
        logger.info(f"Model file already exists: {file_path}")
        return file_path

    # Check if huggingface-cli is available
    try:
        subprocess.run(
            ["huggingface-cli", "--version"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error(
            "huggingface-cli is not installed. Please install it with: pip install huggingface-hub"
        )
        raise FileNotFoundError(
            "huggingface-cli is not installed. Install it with: pip install huggingface-hub"
        )

    # Download the file
    logger.info(f"Downloading {filename} from {repo_id}...")
    cmd = [
        "huggingface-cli",
        "download",
        repo_id,
        filename,
        "--local-dir",
        str(local_dir),
    ]

    if subfolder:
        cmd.extend(["--local-dir-use-symlinks", "False"])

    try:
        subprocess.run(cmd, check=True)
        logger.success(f"Successfully downloaded {filename} to {local_dir}")
        return file_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download {filename} from {repo_id}: {e}")
        raise


def ensure_snow_white_model(
    model_dir: str | Path = "./fairy_tale/models/snow_white_gguf",
) -> Path:
    """
    Ensure the snow_white GGUF model is downloaded and ready.

    This function checks if the model file exists locally, and if not,
    downloads it from HuggingFace automatically.

    Args:
        model_dir: Directory where the model should be stored

    Returns:
        Path to the model file (model-q4_0.gguf)

    Raises:
        FileNotFoundError: If huggingface-cli is not installed
        subprocess.CalledProcessError: If download fails
    """
    model_dir = Path(model_dir)
    model_file = model_dir / "model-q4_0.gguf"

    if model_file.exists():
        logger.debug(f"Snow White model already exists: {model_file}")
        return model_file

    logger.info("Snow White model not found. Downloading from HuggingFace...")
    return download_huggingface_model(
        repo_id="PJiNH/snow_white_gguf",
        filename="model-q4_0.gguf",
        local_dir=model_dir,
    )


def create_ollama_modelfile(
    model_path: Path,
    output_path: Path | None = None,
) -> Path:
    """
    Create an Ollama Modelfile for the GGUF model.

    Args:
        model_path: Path to the GGUF model file
        output_path: Path to save the Modelfile (default: same directory as model)

    Returns:
        Path to the created Modelfile
    """
    if output_path is None:
        output_path = model_path.parent / "Modelfile"

    modelfile_content = """FROM model-q4_0.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER num_predict 200
PARAMETER repeat_penalty 1.1
PARAMETER repeat_last_n 64
"""

    output_path.write_text(modelfile_content, encoding="utf-8")
    logger.info(f"Created Modelfile at {output_path}")
    return output_path


def ensure_ollama_model_registered(
    model_name: str = "snow_white",
    model_dir: str | Path = "./fairy_tale/models/snow_white_gguf",
) -> bool:
    """
    Ensure the Ollama model is registered.

    This function:
    1. Downloads the model if not present
    2. Creates Modelfile if not present
    3. Registers the model with Ollama if not registered

    Args:
        model_name: Name of the Ollama model (default: "snow_white")
        model_dir: Directory containing the model and Modelfile

    Returns:
        True if model is ready, False otherwise
    """
    model_dir = Path(model_dir)

    # 1. Ensure model file exists
    model_file = ensure_snow_white_model(model_dir)

    # 2. Ensure Modelfile exists
    modelfile_path = model_dir / "Modelfile"
    if not modelfile_path.exists():
        create_ollama_modelfile(model_file, modelfile_path)

    # 3. Check if model is registered in Ollama
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        if model_name in result.stdout:
            logger.debug(f"Ollama model '{model_name}' is already registered")
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning(
            "Could not check Ollama model registration. Make sure Ollama is installed and running."
        )
        return False

    # 4. Register the model
    logger.info(f"Registering Ollama model '{model_name}'...")
    try:
        subprocess.run(
            ["ollama", "rm", model_name],
            capture_output=True,
            stderr=subprocess.DEVNULL,
        )  # Ignore error if model doesn't exist
        subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            check=True,
            cwd=str(model_dir),
        )
        logger.success(f"Successfully registered Ollama model '{model_name}'")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to register Ollama model '{model_name}': {e}")
        return False

