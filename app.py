"""
Secure Translation App - Main FastAPI Application
A chat-based translation service with security measures and agentic AI
"""
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import settings
from models.schemas import (
    TranslationRequest, 
    TranslationResponse, 
    ErrorResponse,
    SecurityStatus
)
from agents.security_agent import security_agent
from agents.language_agent import language_agent
from agents.translation_agent import translation_agent
from utils.pdf_processor import pdf_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="Secure Translation API",
    description="A secure, agentic translation service using Mistral AI",
    version="1.0.0"
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Secure Translation API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "api": "operational",
            "mistral_connection": "configured",
            "security_agent": "active",
            "language_agent": "active",
            "translation_agent": "active"
        }
    }

@app.post("/api/translate/text")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def translate_text(request: Request, body: TranslationRequest):
    """
    Translate text to English
    
    - Analyzes input for security threats
    - Detects source language
    - Translates to English
    """
    try:
        text = body.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Step 1: Sanitize input
        sanitized_text = security_agent.sanitize_input(text)
        
        # Step 2: Security analysis
        logger.info("Running security analysis...")
        security_result = await security_agent.analyze(sanitized_text)
        
        if not security_result.is_safe:
            logger.warning(f"Security blocked: {security_result.reason}")
            return TranslationResponse(
                success=False,
                original_text=text[:100] + "..." if len(text) > 100 else text,
                detected_language="N/A",
                detected_language_confidence=0.0,
                translated_text="",
                security_status=security_result.status,
                message=f"Security Alert: {security_result.reason}"
            )
        
        # Step 3: Language detection
        logger.info("Detecting language...")
        language_result = await language_agent.detect(sanitized_text)
        logger.info(f"Detected language: {language_result.language_name} ({language_result.confidence:.2f})")
        
        # Step 4: Translation
        logger.info("Translating text...")
        if language_agent.is_english(language_result):
            translated_text = sanitized_text
            message = "Text is already in English. No translation needed."
        else:
            translated_text = await translation_agent.translate(sanitized_text, language_result)
            message = f"Successfully translated from {language_result.language_name} to English"
        
        return TranslationResponse(
            success=True,
            original_text=text,
            detected_language=language_result.language_name,
            detected_language_confidence=language_result.confidence,
            translated_text=translated_text,
            security_status=SecurityStatus.SAFE,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during translation")

@app.post("/api/translate/pdf")
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def translate_pdf(request: Request, file: UploadFile = File(...)):
    """
    Extract text from PDF and translate to English
    
    - Validates file type and size
    - Extracts text from PDF
    - Analyzes for security threats
    - Detects language and translates
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Validate file
        is_valid, error_message = pdf_processor.validate_file(file_content, file.filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        # Extract text from PDF
        logger.info(f"Extracting text from PDF: {file.filename}")
        try:
            extracted_text = pdf_processor.extract_text(file_content)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        logger.info(f"Extracted {len(extracted_text)} characters from PDF")
        
        # Sanitize extracted text
        sanitized_text = security_agent.sanitize_input(extracted_text)
        
        # Security analysis
        logger.info("Running security analysis on extracted text...")
        security_result = await security_agent.analyze(sanitized_text)
        
        if not security_result.is_safe:
            logger.warning(f"Security blocked PDF content: {security_result.reason}")
            return TranslationResponse(
                success=False,
                original_text=extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text,
                detected_language="N/A",
                detected_language_confidence=0.0,
                translated_text="",
                security_status=security_result.status,
                message=f"Security Alert: {security_result.reason}"
            )
        
        # Language detection
        logger.info("Detecting language of PDF content...")
        language_result = await language_agent.detect(sanitized_text)
        logger.info(f"Detected language: {language_result.language_name}")
        
        # Translation
        logger.info("Translating PDF content...")
        if language_agent.is_english(language_result):
            translated_text = sanitized_text
            message = f"PDF text is already in English. Extracted {len(extracted_text)} characters."
        else:
            translated_text = await translation_agent.translate(sanitized_text, language_result)
            message = f"Successfully translated PDF from {language_result.language_name} to English"
        
        return TranslationResponse(
            success=True,
            original_text=extracted_text,
            detected_language=language_result.language_name,
            detected_language_confidence=language_result.confidence,
            translated_text=translated_text,
            security_status=SecurityStatus.SAFE,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF translation error: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred processing the PDF")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "An unexpected error occurred", "error_code": "INTERNAL_ERROR"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
