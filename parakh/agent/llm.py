"""LLM clients. Each is a function (system, user, json_mode) -> reply text."""

import json
from typing import Any, Callable

import requests

from parakh import config


def _openai_chat(
    url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    json_mode: bool = True,
    extra: dict[str, Any] | None = None,
) -> str:
    """Call an OpenAI-compatible chat completions API and return the text."""
    if not api_key:
        raise RuntimeError(f"No API key set for {model}")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "seed": 42,
        **(extra or {}),
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(url, json=payload, headers=headers, timeout=config.LLM_TIMEOUT_SECONDS)
    if response.status_code == 400 and "response_format" in payload:
        # Some providers/free models reject JSON mode; the prompt already
        # demands JSON and extract_json tolerates surrounding text.
        payload.pop("response_format")
        response = requests.post(url, json=payload, headers=headers, timeout=config.LLM_TIMEOUT_SECONDS)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"].get("content")
    if not content:
        raise RuntimeError(f"{model} returned an empty response")
    return content


def nvidia_chat(system: str, user: str, json_mode: bool = True) -> str:
    """NVIDIA-hosted model (integrate.api.nvidia.com)."""
    return _openai_chat(
        config.NVIDIA_URL, config.NVIDIA_API_KEY, config.NVIDIA_MODEL, system, user, json_mode,
        # Thinking off: keeps content to the JSON decision.
        extra={"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 2048},
    )


def openrouter_chat(system: str, user: str, json_mode: bool = True) -> str:
    return _openai_chat(
        config.OPENROUTER_URL, config.OPENROUTER_API_KEY, config.OPENROUTER_MODEL,
        system, user, json_mode,
    )


def ollama_chat(system: str, user: str, json_mode: bool = True) -> str:
    """Local Ollama, deterministic."""
    payload: dict[str, Any] = {
        "model": config.OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0, "seed": 42},
    }
    if json_mode:
        payload["format"] = "json"
    response = requests.post(
        f"{config.OLLAMA_HOST}/api/chat", json=payload, timeout=config.LLM_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


PROVIDERS: dict[str, Callable[..., str]] = {
    "nvidia": nvidia_chat,
    "openrouter": openrouter_chat,
    "ollama": ollama_chat,
}

# Display name -> client, for the UI dropdown. NVIDIA first = default.
LLMS: dict[str, Callable[..., str]] = {
    f"NVIDIA ({config.NVIDIA_MODEL.split('/')[-1]})": nvidia_chat,
    f"OpenRouter ({config.OPENROUTER_MODEL})": openrouter_chat,
    f"Ollama ({config.OLLAMA_MODEL})": ollama_chat,
}


def default_llm(system: str, user: str, json_mode: bool = True) -> str:
    """Client chosen by LLM_PROVIDER (default nvidia)."""
    return PROVIDERS.get(config.LLM_PROVIDER, nvidia_chat)(system, user, json_mode)


def extract_json(raw: str) -> Any:
    """Parse JSON, tolerating code fences or reasoning text around the object."""
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])
