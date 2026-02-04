"""
Mistral AI API Client Wrapper
Handles all communication with Mistral AI API
"""
import httpx
from typing import List, Dict, Any, Optional
import json
import logging

from config import settings

logger = logging.getLogger(__name__)

class MistralClient:
    """Client for interacting with Mistral AI API"""
    
    def __init__(self):
        self.api_key = settings.MISTRAL_API_KEY
        self.base_url = settings.MISTRAL_BASE_URL
        self.model = settings.MISTRAL_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to Mistral AI
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            response_format: Optional format specification (e.g., {"type": "json_object"})
        
        Returns:
            API response as dictionary
        """
        endpoint = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    endpoint,
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Mistral API: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Mistral API error: {e.response.status_code}")
        except httpx.TimeoutException:
            logger.error("Timeout connecting to Mistral API")
            raise Exception("Request to Mistral API timed out")
        except Exception as e:
            logger.error(f"Error calling Mistral API: {str(e)}")
            raise Exception(f"Error communicating with Mistral AI: {str(e)}")
    
    def extract_response_content(self, response: Dict[str, Any]) -> str:
        """Extract the content from a chat completion response"""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Error extracting response content: {str(e)}")
            raise Exception("Invalid response format from Mistral AI")
    
    async def simple_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3
    ) -> str:
        """
        Simplified completion method with system and user prompts
        
        Returns:
            The content of the assistant's response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self.chat_completion(messages, temperature=temperature)
        return self.extract_response_content(response)
    
    async def json_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Get a JSON response from Mistral AI
        
        Returns:
            Parsed JSON response
        """
        messages = [
            {"role": "system", "content": system_prompt + "\n\nYou must respond with valid JSON only."},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self.chat_completion(
            messages,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        content = self.extract_response_content(response)
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {content}")
            raise Exception("Invalid JSON response from Mistral AI")

# Global client instance
mistral_client = MistralClient()
