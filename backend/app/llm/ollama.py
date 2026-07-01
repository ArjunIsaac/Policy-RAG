import requests
import json


class QwenClient:

    def __init__(self, model: str = "qwen3:8b"):
        self.model = model
        self.url = "http://localhost:11434/api/chat"

    def generate_json(self, prompt: str, context: str):

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self._system_prompt()
                },
                {
                    "role": "user",
                    "content": f"""
Context:
{context}

Task:
{prompt}

Return ONLY valid JSON.
"""
                }
            ],
            "stream": False
        }

        response = requests.post(self.url, json=payload)

        # safety check
        if response.status_code != 200:
            return {"error": "ollama_failed", "raw": response.text}

        data = response.json()

        # FIX: correct ollama structure
        try:
            content = data["message"]["content"]
        except KeyError:
            return {
                "error": "invalid_ollama_response",
                "raw": data
            }

        return self._safe_parse(content)

    def _system_prompt(self):
        return """
You are a strict information extraction engine.

Return ONLY valid JSON.
No explanations.
No markdown.
If missing, use null.
"""

    def _safe_parse(self, text: str):
        try:
            return json.loads(text)
        except Exception:
            return {
                "error": "invalid_json",
                "raw_output": text
            }