import os
import yaml
import torch
from dotenv import load_dotenv

# ------------------------------------------------------------------
# Load environment variables from .env
# ------------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------------
# Load config.yaml from project root (robust to cwd changes)
# ------------------------------------------------------------------
def load_config():
    """
    Loads config.yaml from the project root directory.
    This works regardless of where the script is executed from.
    """
    # api/config.py -> api -> project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Load configuration once at startup
CONFIG = load_config()

# ------------------------------------------------------------------
# Device selection logic
# ------------------------------------------------------------------
def get_device(cfg):
    """
    Selects device based on config and availability.

    system.device:
      - "cpu"   -> force CPU
      - "cuda"  -> force GPU
      - "auto"  -> GPU if available else CPU
    """
    mode = cfg["system"]["device"]

    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        return "cuda"

    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = get_device(CONFIG)

# ------------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------------
API_KEY = os.getenv("API_KEY")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
