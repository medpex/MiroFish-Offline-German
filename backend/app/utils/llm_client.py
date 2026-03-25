"""
LLM-Client-Wrapper
Einheitliche OpenAI-Format-API-Aufrufe
Unterstützt Ollama num_ctx-Parameter zur Vermeidung von Prompt-Abschneidung
"""

import json
import os
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config


class LLMClient:
    """LLM-Client"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY nicht konfiguriert")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

        # Ollama-Kontextfenstergröße — verhindert Prompt-Abschneidung.
        # Wird aus Umgebungsvariable OLLAMA_NUM_CTX gelesen, Standard 8192 (Ollama-Standard ist nur 2048).
        self._num_ctx = int(os.environ.get('OLLAMA_NUM_CTX', '8192'))

    def _is_ollama(self) -> bool:
        """Prüfen, ob wir mit einem Ollama-Server kommunizieren."""
        return '11434' in (self.base_url or '')

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        Chat-Anfrage senden

        Args:
            messages: Nachrichtenliste
            temperature: Temperaturparameter
            max_tokens: Maximale Token-Anzahl
            response_format: Antwortformat (z.B. JSON-Modus)

        Returns:
            Modellantworttext
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        # Für Ollama: num_ctx über extra_body übergeben, um Prompt-Abschneidung zu verhindern
        if self._is_ollama() and self._num_ctx:
            kwargs["extra_body"] = {
                "options": {"num_ctx": self._num_ctx}
            }

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # Einige Modelle (wie MiniMax M2.5) enthalten <think>-Denkinhalte in der Antwort, diese müssen entfernt werden
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Chat-Anfrage senden und JSON zurückgeben

        Args:
            messages: Nachrichtenliste
            temperature: Temperaturparameter
            max_tokens: Maximale Token-Anzahl

        Returns:
            Geparste JSON-Objekt
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # Markdown-Codeblock-Markierungen bereinigen
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            # Versuche ein eingebettetes JSON-Objekt in der Antwort zu finden
            match = re.search(r'\{[\s\S]*\}', cleaned_response)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Ungültiges JSON-Format vom LLM: {cleaned_response}")
