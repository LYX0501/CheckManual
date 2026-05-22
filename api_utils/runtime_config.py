import json
import os
from pathlib import Path


CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"
CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"


def _read_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _read_toml(path):
    if not path.exists():
        return {}
    data = {}
    current_section = None
    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section_name = line[1:-1].strip()
                section_parts = section_name.split(".")
                current_section = data
                for part in section_parts:
                    part = part.strip().strip('"')
                    current_section = current_section.setdefault(part, {})
                continue
            if "=" not in line:
                continue
            key, value = [item.strip() for item in line.split("=", 1)]
            target = current_section if current_section is not None else data
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.lower() in {"true", "false"}:
                value = value.lower() == "true"
            target[key] = value
    return data


def _normalize_chat_url(url):
    url = (url or "").strip().rstrip("/")
    if not url:
        return ""
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/responses"):
        return url[: -len("/responses")] + "/chat/completions"
    return url + "/chat/completions"


def load_codex_openai_config():
    auth = _read_json(CODEX_AUTH_PATH)
    config = _read_toml(CODEX_CONFIG_PATH)

    api_key = (auth.get("OPENAI_API_KEY") or "").strip()
    provider_name = config.get("model_provider")
    providers = config.get("model_providers", {})
    provider_cfg = providers.get(provider_name, {}) if provider_name else {}
    wire_api = (provider_cfg.get("wire_api") or "").strip()

    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or provider_cfg.get("base_url")
        or ""
    )
    model = (config.get("model") or "").strip()
    return {
        "key": api_key,
        "chat_url": _normalize_chat_url(base_url),
        "base_url": (base_url or "").strip().rstrip("/"),
        "model": model,
        "wire_api": wire_api,
    }


def resolve_gpt_config(static_config):
    gpt_config = static_config.get("GPT", {}) if isinstance(static_config, dict) else {}
    key = (
        os.environ.get("CHECKMANUAL_GPT_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or gpt_config.get("key", "")
    )
    url = os.environ.get("CHECKMANUAL_GPT_URL") or gpt_config.get("url", "")
    model = os.environ.get("CHECKMANUAL_GPT_MODEL") or gpt_config.get("model", "")

    key = key.strip()
    url = url.strip()
    model = model.strip()

    if key and url:
        return {
            "key": key,
            "chat_url": _normalize_chat_url(url),
            "base_url": url.rstrip("/"),
            "model": model,
            "wire_api": "",
        }

    codex_cfg = load_codex_openai_config()
    if codex_cfg["key"] and codex_cfg["chat_url"]:
        return {
            "key": key or codex_cfg["key"],
            "chat_url": url or codex_cfg["chat_url"],
            "base_url": url or codex_cfg["base_url"],
            "model": model or codex_cfg["model"],
            "wire_api": codex_cfg.get("wire_api", ""),
        }

    return {
        "key": key,
        "chat_url": _normalize_chat_url(url),
        "base_url": url.rstrip("/"),
        "model": model,
        "wire_api": "",
    }


def choose_gpt_model(requested_model, resolved_config):
    requested_model = (requested_model or "").strip()
    fallback_model = (resolved_config.get("model") or "").strip()
    if not requested_model:
        return fallback_model or "gpt-5.4"
    if (
        resolved_config.get("wire_api") == "responses"
        and fallback_model
        and requested_model in {"gpt-4o", "gpt-4o-2024-08-06"}
    ):
        return fallback_model
    return requested_model
