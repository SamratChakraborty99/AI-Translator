"""
Configuration module for the Secure Translation App
Loads API keys and application settings
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Load Mistral API Key from Key.txt
KEY_FILE_PATH = PROJECT_ROOT / "Key.txt"

def load_api_key() -> str:
    """
    Load the Mistral API key.
    Priority:
    1. Environment variable MISTRAL_API_KEY
    2. Key.txt file (for local development)
    """
    # First, check environment variable
    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key:
        return api_key
    
    # Fall back to Key.txt for local development
    try:
        with open(KEY_FILE_PATH, 'r') as f:
            api_key = f.read().strip()
            if not api_key:
                raise ValueError("API key is empty")
            return api_key
    except FileNotFoundError:
        raise FileNotFoundError(
            f"API key not found. Either set MISTRAL_API_KEY environment variable "
            f"or create Key.txt at {KEY_FILE_PATH}"
        )
    except Exception as e:
        raise Exception(f"Error loading API key: {str(e)}")

# Application Settings
class Settings:
    # API Configuration
    MISTRAL_API_KEY: str = load_api_key()
    MISTRAL_BASE_URL: str = "https://api.mistral.ai/v1"
    MISTRAL_MODEL: str = "mistral-small-latest"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    
    # File Upload Settings
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_FILE_TYPES: list = [".pdf"]
    
    # Security Settings
    MAX_INPUT_LENGTH: int = 50000  # Maximum characters for text input
    BLOCKED_PATTERNS: list = [
        "ignore previous instructions",
        "ignore all instructions",
        "disregard your instructions",
        "forget your instructions",
        "system prompt",
        "reveal your prompt",
        "show your instructions",
    ]
    
    # CORS Settings
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]

settings = Settings()
