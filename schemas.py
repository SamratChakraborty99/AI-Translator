"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class SecurityStatus(str, Enum):
    SAFE = "safe"
    BLOCKED = "blocked"
    WARNING = "warning"

class TranslationRequest(BaseModel):
    """Request schema for translation"""
    text: str = Field(..., min_length=1, max_length=50000, description="Text to translate")
    
class TranslationResponse(BaseModel):
    """Response schema for translation"""
    success: bool
    original_text: str
    detected_language: str
    detected_language_confidence: float
    translated_text: str
    security_status: SecurityStatus
    message: Optional[str] = None

class SecurityAnalysisResult(BaseModel):
    """Result of security analysis"""
    is_safe: bool
    status: SecurityStatus
    reason: Optional[str] = None
    risk_score: float = Field(ge=0, le=1)

class LanguageDetectionResult(BaseModel):
    """Result of language detection"""
    language_code: str
    language_name: str
    confidence: float = Field(ge=0, le=1)

class ErrorResponse(BaseModel):
    """Error response schema"""
    success: bool = False
    error: str
    error_code: str

class ChatMessage(BaseModel):
    """Chat message for conversation history"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None

class ConversationHistory(BaseModel):
    """Conversation history for context"""
    messages: List[ChatMessage] = []
