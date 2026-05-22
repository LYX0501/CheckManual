import json
from pathlib import Path
from types import SimpleNamespace

import requests
from api_utils.runtime_config import choose_gpt_model, resolve_gpt_config


def _load_api_config():
    config_path = Path(__file__).resolve().parents[1] / "api_utils" / "api_key_config.json"
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    resolved = resolve_gpt_config(config)
    gpt_key = resolved["key"]
    gpt_url = resolved["chat_url"]
    gpt_model = resolved["model"]

    if not gpt_key or not gpt_url:
        raise RuntimeError(
            "Missing GPT configuration. Set CHECKMANUAL_GPT_KEY and CHECKMANUAL_GPT_URL, "
            f"fill {config_path}, or rely on ~/.codex/auth.json + ~/.codex/config.toml."
        )

    config["GPT"] = {
        "key": gpt_key,
        "url": gpt_url,
        "base_url": resolved["base_url"],
        "model": gpt_model,
        "wire_api": resolved.get("wire_api", ""),
    }
    return config


def _extract_responses_text(data):
    texts = []
    for item in data.get("output", []):
        for entry in item.get("content", []):
            if entry.get("type") == "output_text":
                texts.append(entry.get("text", ""))
    return "".join(texts)


def _post_chat_completions(api_config, headers, model, messages, temperature=0.1, response_format=None):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    response = requests.post(
        api_config["GPT"]["url"],
        json=payload,
        headers=headers,
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


class _ChatCompletions:
    def __init__(self, api_config):
        self._api_config = api_config

    def create(self, model, messages, temperature=0.1, response_format=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": self._api_config["GPT"]["key"],
        }
        request_model = choose_gpt_model(model, self._api_config["GPT"])
        if self._api_config["GPT"].get("wire_api") == "responses":
            payload = {
                "model": request_model,
                "input": messages,
                "temperature": temperature,
                "stream": False,
            }
            response = requests.post(
                self._api_config["GPT"]["base_url"] + "/responses",
                json=payload,
                headers=headers,
                timeout=180,
            )
            if response.status_code >= 500:
                content = _post_chat_completions(
                    self._api_config,
                    headers,
                    request_model,
                    messages,
                    temperature=temperature,
                    response_format=response_format,
                )
            else:
                response.raise_for_status()
                data = response.json()
                content = _extract_responses_text(data)
        else:
            content = _post_chat_completions(
                self._api_config,
                headers,
                request_model,
                messages,
                temperature=temperature,
                response_format=response_format,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class GPT:
    def __init__(self) -> None:
        api_config = _load_api_config()
        self.default_model = api_config["GPT"].get("model") or None
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=_ChatCompletions(api_config))
        )
