import json
import os
import requests
import base64
from pathlib import Path
from api_utils.runtime_config import choose_gpt_model, resolve_gpt_config

CONFIG_PATH = Path(__file__).resolve().with_name("api_key_config.json")

with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
    api_key_config = json.load(file)


def _load_gpt_config():
    resolved = resolve_gpt_config(api_key_config)
    key = resolved["key"]
    url = resolved["chat_url"]
    if not key or not url:
        raise RuntimeError(
            "Missing GPT configuration. Set CHECKMANUAL_GPT_KEY and CHECKMANUAL_GPT_URL, "
            f"fill {CONFIG_PATH}, or rely on ~/.codex/auth.json + ~/.codex/config.toml."
        )
    return resolved


def _extract_responses_text(response_json):
    texts = []
    for item in response_json.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    return "".join(texts)


def _convert_chat_messages_to_responses_input(messages):
    converted_messages = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, str):
            converted_messages.append(
                {
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                }
            )
            continue

        if isinstance(content, list):
            converted_content = []
            for item in content:
                item_type = item.get("type")
                if item_type == "text":
                    converted_content.append(
                        {
                            "type": "input_text",
                            "text": item.get("text", ""),
                        }
                    )
                elif item_type == "image_url":
                    image_url = item.get("image_url", {})
                    converted_content.append(
                        {
                            "type": "input_image",
                            "image_url": image_url.get("url", ""),
                            "detail": image_url.get("detail", "auto"),
                        }
                    )
                else:
                    converted_content.append(item)
            converted_messages.append({"role": role, "content": converted_content})
            continue

        converted_messages.append({"role": role, "content": content})

    return converted_messages


def _post_chat_completions(resolved, header, model, prompt, timeout=180):
    post_dict = {
        "model": model,
        "messages": prompt,
        "stream": False,
    }
    response = requests.post(resolved["chat_url"], json=post_dict, headers=header, timeout=timeout)
    response.raise_for_status()
    json_r = response.json()
    return json_r["choices"][0]["message"]["content"]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def gpt_response(prompt, model_version='gpt-4o'):
    message = ""
    for idx in range(100):
        try:
            # print("read GPT key")
            resolved = _load_gpt_config()
            key = resolved["key"]
            model = choose_gpt_model(model_version, resolved)
            header = {
                "Content-Type":"application/json",
                "Authorization": key
            }
            print("Try request GPT")
            if resolved.get("wire_api") == "responses":
                post_dict = {
                    "model": model,
                    "input": _convert_chat_messages_to_responses_input(prompt),
                    "stream": False,
                }
                r = requests.post(
                    resolved["base_url"] + "/responses",
                    json=post_dict,
                    headers=header,
                    timeout=180,
                )
                if r.status_code >= 500:
                    message = _post_chat_completions(resolved, header, model, prompt)
                else:
                    r.raise_for_status()
                    json_r = r.json()
                    message = _extract_responses_text(json_r)
            else:
                message = _post_chat_completions(resolved, header, model, prompt)
            break
        except KeyboardInterrupt:
            print("Stop")
            break
        except Exception as e:
            print(e)
            # print(r.json())
            pass
    if not message:
        raise RuntimeError("GPT request failed after retries.")
    # price = r['usage']['completion_tokens']/1000*0.43 + r['usage']['prompt_tokens']/1000*0.22
    # print("gpt4 used time: %.2f, used price: %.5f"%(time.time()-start_time,price))
    return message

def gptv_response(prompt, model_version='gpt-4o'):
    message = ""
    for idx in range(100):
        try:
            resolved = _load_gpt_config()
            key = resolved["key"]
            model = choose_gpt_model(model_version, resolved)
            header = {
                "Content-Type":"application/json",
                "Authorization": key
            }
            if resolved.get("wire_api") == "responses":
                post_dict = {
                    "model": model,
                    "input": _convert_chat_messages_to_responses_input(prompt),
                    "stream": False,
                }
                r = requests.post(
                    resolved["base_url"] + "/responses",
                    json=post_dict,
                    headers=header,
                    timeout=180,
                )
                if r.status_code >= 500:
                    message = _post_chat_completions(resolved, header, model, prompt)
                else:
                    r.raise_for_status()
                    json_r = r.json()
                    message = _extract_responses_text(json_r)
            else:
                message = _post_chat_completions(resolved, header, model, prompt)
            break
        except KeyboardInterrupt:
            print("Stop")
            break
        except Exception as e:
            print(e)
            # print(json_r)
            # print(r.json())
            continue
    if not message:
        raise RuntimeError("GPT-V request failed after retries.")
    return message
