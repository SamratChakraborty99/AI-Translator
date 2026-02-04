"""
Language Detection Agent
Detects the language of input text using Mistral AI
"""
import logging
from typing import Optional

from utils.mistral_client import mistral_client
from models.schemas import LanguageDetectionResult

logger = logging.getLogger(__name__)

class LanguageAgent:
    """
    Language Detection Agent that identifies the source language of text
    """
    
    SYSTEM_PROMPT = """You are a language detection expert. Your task is to identify the language of the provided text.

Analyze the text and respond with a JSON object containing:
{
    "language_code": "ISO 639-1 two-letter code (e.g., 'en', 'fr', 'es', 'de', 'zh', 'ja', 'ar', 'hi')",
    "language_name": "Full name of the language in English (e.g., 'French', 'Spanish', 'German')",
    "confidence": float between 0 and 1 indicating how confident you are
}

If the text contains multiple languages, identify the predominant one.
If the text is already in English, still identify it as English.
Be accurate and confident in your detection."""

    # Common language mappings for fallback
    LANGUAGE_NAMES = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'bn': 'Bengali',
        'tr': 'Turkish',
        'nl': 'Dutch',
        'pl': 'Polish',
        'sv': 'Swedish',
        'no': 'Norwegian',
        'da': 'Danish',
        'fi': 'Finnish',
        'el': 'Greek',
        'he': 'Hebrew',
        'th': 'Thai',
        'vi': 'Vietnamese',
        'id': 'Indonesian',
        'ms': 'Malay',
        'tl': 'Tagalog',
        'uk': 'Ukrainian',
        'cs': 'Czech',
        'ro': 'Romanian',
        'hu': 'Hungarian',
    }

    async def detect(self, text: str) -> LanguageDetectionResult:
        """
        Detect the language of the provided text
        
        Args:
            text: Text to analyze
        
        Returns:
            LanguageDetectionResult with language info
        """
        # Take a sample of the text for detection (first 1000 chars is usually enough)
        sample_text = text[:1000] if len(text) > 1000 else text
        
        try:
            user_prompt = f"""Detect the language of the following text:

TEXT:
---
{sample_text}
---

Respond with JSON only."""

            response = await mistral_client.json_completion(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1
            )
            
            language_code = response.get("language_code", "unknown").lower()
            language_name = response.get("language_name", "Unknown")
            confidence = float(response.get("confidence", 0.8))
            
            # Validate and normalize
            if language_code in self.LANGUAGE_NAMES:
                language_name = self.LANGUAGE_NAMES[language_code]
            
            return LanguageDetectionResult(
                language_code=language_code,
                language_name=language_name,
                confidence=min(max(confidence, 0), 1)  # Clamp between 0 and 1
            )
            
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            # Default to unknown if detection fails
            return LanguageDetectionResult(
                language_code="unknown",
                language_name="Unknown",
                confidence=0.0
            )
    
    def is_english(self, result: LanguageDetectionResult) -> bool:
        """Check if the detected language is English"""
        return result.language_code.lower() == 'en'

# Global instance
language_agent = LanguageAgent()
