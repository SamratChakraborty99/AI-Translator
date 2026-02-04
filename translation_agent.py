"""
Translation Agent
Translates text from any language to English using Mistral AI
"""
import logging
from typing import Optional

from utils.mistral_client import mistral_client
from models.schemas import LanguageDetectionResult

logger = logging.getLogger(__name__)

class TranslationAgent:
    """
    Translation Agent that translates text to English
    """
    
    SYSTEM_PROMPT = """You are a professional translator. Your task is to translate text accurately to English.

Guidelines:
1. Translate the text accurately while preserving the original meaning
2. Maintain the original formatting (paragraphs, lists, etc.) where possible
3. Keep proper nouns unchanged unless they have common English equivalents
4. Preserve numbers, dates, and technical terms appropriately
5. If the text contains phrases that don't translate directly, provide the closest English equivalent
6. Maintain the tone and style of the original text

Only provide the translated text in your response, nothing else.
Do not add explanations, notes, or commentary.
Do not say "Here is the translation" or similar phrases."""

    # Maximum characters per chunk for long texts
    MAX_CHUNK_SIZE = 4000
    
    async def translate(
        self,
        text: str,
        source_language: Optional[LanguageDetectionResult] = None
    ) -> str:
        """
        Translate text to English
        
        Args:
            text: Text to translate
            source_language: Optional language detection result
        
        Returns:
            Translated text in English
        """
        # If text is already in English, return as-is
        if source_language and source_language.language_code == 'en':
            logger.info("Text is already in English, no translation needed")
            return text
        
        # For long texts, translate in chunks
        if len(text) > self.MAX_CHUNK_SIZE:
            return await self._translate_chunked(text, source_language)
        
        return await self._translate_single(text, source_language)
    
    async def _translate_single(
        self,
        text: str,
        source_language: Optional[LanguageDetectionResult] = None
    ) -> str:
        """Translate a single chunk of text"""
        try:
            # Build the user prompt
            if source_language:
                lang_info = f"The source language is {source_language.language_name} ({source_language.language_code})."
            else:
                lang_info = "Detect the source language automatically."
            
            user_prompt = f"""{lang_info}

Translate the following text to English:

---
{text}
---

Provide only the English translation, nothing else."""

            translated = await mistral_client.simple_completion(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.2
            )
            
            return translated.strip()
            
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            raise Exception(f"Translation failed: {str(e)}")
    
    async def _translate_chunked(
        self,
        text: str,
        source_language: Optional[LanguageDetectionResult] = None
    ) -> str:
        """Translate long text in chunks"""
        logger.info(f"Translating long text ({len(text)} chars) in chunks")
        
        # Split text into chunks at sentence boundaries
        chunks = self._split_into_chunks(text)
        translated_chunks = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Translating chunk {i+1}/{len(chunks)}")
            translated = await self._translate_single(chunk, source_language)
            translated_chunks.append(translated)
        
        return "\n\n".join(translated_chunks)
    
    def _split_into_chunks(self, text: str) -> list:
        """Split text into manageable chunks at sentence boundaries"""
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            # If adding this paragraph would exceed limit, save current chunk
            if len(current_chunk) + len(para) > self.MAX_CHUNK_SIZE and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # If a single paragraph is too long, split by sentences
            if len(para) > self.MAX_CHUNK_SIZE:
                sentences = self._split_into_sentences(para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > self.MAX_CHUNK_SIZE and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    current_chunk += sentence + " "
            else:
                current_chunk += para + "\n\n"
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _split_into_sentences(self, text: str) -> list:
        """Simple sentence splitter"""
        import re
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

# Global instance
translation_agent = TranslationAgent()
