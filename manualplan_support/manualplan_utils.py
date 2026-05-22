import json
import os
import re


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_results_root():
    return ensure_dir(os.path.join(ROOT_DIR, "results"))


def normalize_out_dir(out_dir):
    results_root = os.path.abspath(get_results_root())
    root_dir = os.path.abspath(ROOT_DIR)

    if not out_dir:
        return results_root

    if not os.path.isabs(out_dir):
        parent = os.path.dirname(out_dir)
        base_name = os.path.basename(out_dir)
        if parent in ("", ".") and (base_name == "results" or base_name.startswith("results_")):
            if base_name == "results":
                return results_root
            return os.path.join(results_root, base_name)

    abs_out_dir = os.path.abspath(out_dir)
    if abs_out_dir == results_root or abs_out_dir.startswith(results_root + os.sep):
        return abs_out_dir

    if os.path.dirname(abs_out_dir) == root_dir:
        base_name = os.path.basename(abs_out_dir)
        if base_name == "results":
            return results_root
        if base_name.startswith("results_"):
            return os.path.join(results_root, base_name)

    return abs_out_dir


def normalize_results_file(path, default_path):
    default_path = os.path.abspath(default_path)
    default_dir = ensure_dir(os.path.dirname(default_path))
    results_root = os.path.abspath(get_results_root())
    root_dir = os.path.abspath(ROOT_DIR)

    if not path:
        return default_path

    if not os.path.isabs(path):
        parent = os.path.dirname(path)
        if parent in ("", "."):
            return os.path.join(default_dir, os.path.basename(path))

    abs_path = os.path.abspath(path)
    if abs_path == results_root:
        return default_path
    if abs_path.startswith(results_root + os.sep):
        return abs_path

    if os.path.dirname(abs_path) == root_dir:
        return os.path.join(default_dir, os.path.basename(abs_path))

    return abs_path


def get_object_urdf_path(shape_id, data_dir):
    return os.path.join(data_dir, str(shape_id), "mobility.urdf")


def get_semantics_path(shape_id, data_dir):
    return os.path.join(data_dir, str(shape_id), "semantics.txt")


def get_robot_gripper_urdf_path():
    return os.path.join(ROOT_DIR, "assets", "robot", "panda", "panda_gripper.urdf")


def parse_manual_pdf_metadata(pdf_file_name):
    manual_stem = os.path.splitext(os.path.basename(pdf_file_name))[0]
    match = re.match(
        r"(?P<shape_id>\d+)_(?P<category>.+?)_manual(?:_(?P<group>group\d+))?$",
        manual_stem,
    )
    if not match:
        return {
            "shape_id": manual_stem.split("_", 1)[0],
            "category": manual_stem,
            "group_name": None,
            "group_index": None,
            "manual_stem": manual_stem,
            "manual_pdf": os.path.basename(pdf_file_name),
        }
    group_name = match.group("group")
    group_index = None
    if group_name:
        group_match = re.search(r"(\d+)$", group_name)
        if group_match:
            group_index = int(group_match.group(1))
    return {
        "shape_id": match.group("shape_id"),
        "category": match.group("category"),
        "group_name": group_name,
        "group_index": group_index,
        "manual_stem": manual_stem,
        "manual_pdf": os.path.basename(pdf_file_name),
    }


def infer_sample_metadata(sample_dir):
    manual_pdf = find_manual_pdf(sample_dir)
    metadata = parse_manual_pdf_metadata(manual_pdf)
    metadata["sample_dir"] = sample_dir
    return metadata


def list_manual_subdirs(manual_dir, sample=None, max_samples=None):
    manual_subdirs = sorted(
        subdir
        for subdir in os.listdir(manual_dir)
        if os.path.isdir(os.path.join(manual_dir, subdir)) and not subdir.startswith(".")
    )

    if sample:
        matched = [
            subdir
            for subdir in manual_subdirs
            if subdir == sample or subdir.startswith(f"{sample}_")
        ]
        if not matched:
            raise FileNotFoundError(
                f"Cannot find sample '{sample}' under manual_dir: {manual_dir}"
            )
        manual_subdirs = matched[:1]

    if max_samples is not None:
        manual_subdirs = manual_subdirs[:max_samples]

    return manual_subdirs


def find_manual_pdf(sample_dir):
    pdf_files = sorted(
        file_name for file_name in os.listdir(sample_dir) if file_name.endswith(".pdf")
    )
    if not pdf_files:
        raise FileNotFoundError(f"No manual PDF found in sample dir: {sample_dir}")
    return pdf_files[0]


def find_part_state_file(sample_dir):
    candidates = sorted(
        file_name
        for file_name in os.listdir(sample_dir)
        if file_name.endswith("functions.json")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No part-state json ending with 'functions.json' found in: {sample_dir}"
        )
    return candidates[0]


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_semantics(shape_id, data_dir):
    semantic_file = get_semantics_path(shape_id, data_dir)
    if not os.path.exists(semantic_file):
        raise FileNotFoundError(f"Missing semantics file: {semantic_file}")

    link_dict = {}
    with open(semantic_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(" ", 1)
            link_dict[key] = value
    return link_dict


def load_task_dict(sample_dir, global_tasks_path=None, global_key=None):
    local_candidates = []

    preferred_local_files = [
        "eval_tasks.json",
        "tasks.json",
    ]
    for file_name in preferred_local_files:
        json_path = os.path.join(sample_dir, file_name)
        if os.path.exists(json_path):
            local_candidates.append(json_path)

    for file_name in sorted(os.listdir(sample_dir)):
        if not file_name.endswith("tasks.json"):
            continue
        json_path = os.path.join(sample_dir, file_name)
        if json_path not in local_candidates:
            local_candidates.append(json_path)

    for json_path in local_candidates:
        task_dict = load_json(json_path)
        if isinstance(task_dict, dict):
            return task_dict, json_path

    if global_tasks_path:
        global_task_dict = load_json(global_tasks_path)
        if global_key and global_key in global_task_dict:
            return global_task_dict[global_key], global_tasks_path
        if len(global_task_dict) == 1:
            only_value = next(iter(global_task_dict.values()))
            if isinstance(only_value, dict):
                return only_value, global_tasks_path

    raise FileNotFoundError(
        f"Cannot find a usable task json in {sample_dir}"
        + (
            f" or global task file {global_tasks_path}"
            if global_tasks_path
            else ""
        )
    )
