import argparse
import os
import re

import requests


DEFAULT_TIMEOUT_SECONDS = 120


GROUP_TO_PORT = {
    "group1": 5001,
    "group2": 5002,
    "group3": 5003,
}


def infer_group_name(img_path, group_name=None):
    if group_name:
        normalized = group_name.strip().lower()
        if normalized in GROUP_TO_PORT:
            return normalized

    normalized_path = img_path.lower()
    for candidate in GROUP_TO_PORT:
        if candidate in normalized_path:
            return candidate

    sample_dir = os.path.dirname(os.path.abspath(img_path))
    try:
        for file_name in os.listdir(sample_dir):
            if not file_name.lower().endswith(".pdf"):
                continue
            match = re.search(r"(group\d+)", file_name.lower())
            if match and match.group(1) in GROUP_TO_PORT:
                return match.group(1)
    except OSError:
        pass

    raise ValueError(
        f"Unable to infer CV server group for image path: {img_path}. "
        "Please pass --group group1|group2|group3."
    )


def call_cv_server(img_path, category, group_name=None, timeout=DEFAULT_TIMEOUT_SECONDS):
    port_override = os.environ.get("CHECKMANUAL_CV_SERVER_PORT", "").strip()
    if port_override:
        port_num = int(port_override)
    else:
        resolved_group = infer_group_name(img_path, group_name=group_name)
        port_num = GROUP_TO_PORT[resolved_group]
    cv_server_url = f"http://localhost:{port_num}/sam"
    data = {"img_path": img_path, "category": category}
    print(data)
    try:
        response = requests.post(cv_server_url, json=data, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"CV server request timed out after {timeout}s: {cv_server_url}. "
            "Start perception/cv_server.py or set CHECKMANUAL_CV_SERVER_PORT."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"CV server request failed: {cv_server_url}. "
            "Start perception/cv_server.py or set CHECKMANUAL_CV_SERVER_PORT."
        ) from exc
    print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send image path to CV server.")
    parser.add_argument("img_path", type=str, help="Path to the image")
    parser.add_argument("category", type=str, help="appliance category")
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Manual group name, e.g. group1/group2/group3",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="CV server request timeout in seconds.",
    )
    args = parser.parse_args()

    call_cv_server(args.img_path, args.category, group_name=args.group, timeout=args.timeout)
