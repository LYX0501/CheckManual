import math
import os
import pickle
import re
import subprocess
import sys

import numpy as np
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm

from api_utils.gpt_api import encode_image, gpt_response, gptv_response
from track1_ManualPlan import get_camera_angles, resolve_manual, vis_analyze_page
from manualplan_support.manualplan_utils import (
    ROOT_DIR,
    ensure_dir,
    find_manual_pdf,
    get_object_urdf_path,
    get_robot_gripper_urdf_path,
    list_manual_subdirs,
    load_json,
    load_semantics,
    load_task_dict,
    normalize_out_dir,
    normalize_results_file,
)


total_planning = 0
success_planning = 0
total_execution = 0
success_execution = 0


def load_init_pose(sample_dir):
    pred_obj_pose_path = os.path.join(sample_dir, "pred_obj_pose.json")
    if os.path.exists(pred_obj_pose_path):
        return load_json(pred_obj_pose_path)
    return {"init_pos": [0, 0, 0], "init_rot": [1, 0, 0, 0]}


def build_env_and_camera(args, shape_id, init_pose, obj_init_scale, view, img_size, flog):
    from env import Env
    from camera import Camera

    env = Env(flog=flog, show_gui=args.gui)
    object_urdf_fn = get_object_urdf_path(shape_id, args.data_dir)
    flog.write(f"object_urdf_fn: {object_urdf_fn}\n")
    object_material = env.get_material(4, 4, 0.01)
    flog.write("Object State: closed\n")
    env.load_object(
        object_urdf_fn,
        object_material,
        state="closed",
        init_scale=obj_init_scale,
        init_pos=init_pose["init_pos"],
        init_rot=init_pose["init_rot"],
    )

    phi, theta = get_camera_angles(view)
    cam = Camera(env, image_size=img_size, dist=3.0, phi=phi, theta=theta, fixed_position=True)
    return env, cam


def build_robot(env, robot_urdf_path):
    from robots.panda_robot_gripper import Robot

    robot_material = env.get_material(4, 4, 0.01)
    return Robot(env, robot_urdf_path, robot_material, open_gripper=True, scale=1.0)


def align_parts_nocad(
    sample_dir,
    manual_subdir,
    page_vis_info_dict,
    use_cache=True,
    env=None,
    cam=None,
):
    link_pngs_dir = ensure_dir(os.path.join(sample_dir, "link_pngs"))
    link_function_json = os.path.join(link_pngs_dir, "pred_mask_function_nocad.json")
    if os.path.exists(link_function_json) and use_cache:
        return load_json(link_function_json)

    if env is None or cam is None:
        raise RuntimeError(
            "pred_mask_function_nocad.json is missing, but no environment/camera was "
            "provided to regenerate it."
        )

    page_vis_info = ""
    for page_file, page_content in page_vis_info_dict.items():
        page_idx = page_file.replace(".png", "")
        page_vis_info += f"Page {page_idx}: {page_content} | \n"

    prompt = [
        {
            "role": "system",
            "content": "I have leveraged multimodal large model to analyze each page of an appliance manual. "
            "Please tell me which pages display figures about introducing appliance components overview and control panel. "
            "Your answer should be a dictionary, in which each item takes page number as key and corresponding figure "
            "information as value. You can ignore the first page.",
        },
        {"role": "user", "content": f"Appliance Manual Content:\n{page_vis_info}"},
    ]

    figure_page_idxs = []
    while True:
        response = gpt_response(prompt)
        key_value_pattern = re.compile(r'(\d+):\s*"([^"]+)"')
        matches = key_value_pattern.findall(response)
        resolved_dict = {int(key): value for key, value in matches}
        figure_page_idxs = sorted(resolved_dict.keys())[:3]
        if figure_page_idxs:
            break

    figure_pages = [
        os.path.join(sample_dir, "manual_pngs", f"{page_idx}.png")
        for page_idx in figure_page_idxs
    ]

    env.step()
    env.render()
    rgb, _ = cam.get_observation()
    marked_rgb = (rgb * 255).astype(np.uint8)
    img_path = os.path.join(sample_dir, "rgb_track3.png")
    Image.fromarray(marked_rgb).save(img_path)

    category = "_".join(manual_subdir.split("_")[1:])
    if category and category[-1].isdigit():
        category = "_".join(category.split("_")[:-1])

    command = [
        sys.executable,
        os.path.join(ROOT_DIR, "perception", "call_cv_server_sam.py"),
        img_path,
        category,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "call_cv_server_sam.py failed")

    cropped_rgb_masked_ids_path = os.path.join(sample_dir, "cropped_rgb_masked_ids_track3.png")
    encoded_link_img = encode_image(cropped_rgb_masked_ids_path)
    text_prompt = (
        "I will provide you with an appliance photo with segmentation mask IDs and several page screenshots "
        "from the appliance manual. On the appliance photo, components are annotated with segmentation mask IDs. "
        "A part of components are labeled with function names in the manual. Please analyze every diagrams in the "
        "given manual pages and identify the relationships between mask IDs and function names. Your answer should "
        "be a dictionary which takes mask ID as key and the function name as value."
    )
    prompt_content = [
        {"type": "text", "text": text_prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_link_img}", "detail": "high"},
        },
    ]

    link_function_alignment = {}
    while True:
        copied_prompt_content = list(prompt_content)
        for figure_page in figure_pages:
            encoded_page_img = encode_image(figure_page)
            copied_prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_page_img}",
                        "detail": "high",
                    },
                }
            )
        response = gptv_response([{"role": "user", "content": copied_prompt_content}])
        matches = re.findall(r'(\d+):\s*"([^"]+)"', response)
        link_function_alignment = {value: int(key) for key, value in matches}
        if link_function_alignment:
            break

    with open(link_function_json, "w", encoding="utf-8") as file:
        json.dump(link_function_alignment, file, ensure_ascii=False, indent=4)

    return link_function_alignment


def track3_plan_steps(
    sample_dir,
    task_idx,
    manual_content,
    task_name,
    pred_link_function_dict,
    llm_version="gpt-4o",
    use_cache_plan=True,
):
    task_save_path = os.path.join(sample_dir, f"track3_task_{task_idx}_plans.json")
    if os.path.exists(task_save_path) and use_cache_plan:
        task_plan = load_json(task_save_path)
        return task_plan["task_plan"]

    prompt = [
        {
            "role": "system",
            "content": "I will provide you with an appliance manual and a list of component function names. "
            "Please help me plan how to complete the given task. If the task is metioned in the manual, you can "
            "follow the manual content. Otherwise, you should combine common sense and appliance part functions "
            "to make plan. Every planing step should follow 'Function name: Operation' like 'Power Button: Press 1 time'. "
            "Candidate operations only include ['Press ... times', 'Rotate ... degrees', 'Open', 'Close']. "
            "Your answer must be a python list sequenced from step 1 to step N.",
        },
        {
            "role": "user",
            "content": (
                f"Task: {task_name} ; Appliance Manual:\n{manual_content} ; "
                f"Component Function Name List: {pred_link_function_dict}"
            ),
        },
    ]

    planned_steps = []
    while True:
        response = gpt_response(prompt, model_version=llm_version)
        response = response.replace('"', "'").replace("*", "")
        matches = re.findall(r"'([^']+):\s*([^']+)'", response)
        planned_steps = [f"{match[0]}: {match[1]}" for match in matches]
        if "\n" not in str(planned_steps) and "python" not in str(planned_steps) and planned_steps:
            break

    with open(task_save_path, "w", encoding="utf-8") as file:
        json.dump({"task_plan": planned_steps}, file, ensure_ascii=False, indent=4)

    return planned_steps


def normalize_track3_gt_steps(gt_steps):
    return [
        (" ".join(item[1].split()[1:])).split(" (")[0] + ": " + item[2]
        for item in gt_steps
    ]


def mask2bbox(mask):
    nonzero_indices = np.nonzero(mask)
    if len(nonzero_indices[0]) == 0:
        raise ValueError("target part mask is empty")
    min_y, min_x = np.min(nonzero_indices, axis=1)
    max_y, max_x = np.max(nonzero_indices, axis=1)
    return [int(min_x), int(min_y), int(max_x), int(max_y)]


def get_mask_item(all_masks, mask_id):
    if isinstance(all_masks, dict):
        return all_masks.get(mask_id) or all_masks.get(str(mask_id))
    return all_masks[mask_id]


def vlm_plan_execution(env, cam, robot, target_part_function_name, target_part_mask, instruction, task_dir):
    from manualplan_support.api_omnimanip import EnvInterface
    from voxposer.run import run_voxposer

    my_robot = EnvInterface(env, robot, cam, task_dir=task_dir)
    observation = my_robot.get_observation()
    observation["detect_info"] = {
        "boxes": [mask2bbox(target_part_mask)],
        "masks": {0: target_part_mask},
        "categorys": [target_part_function_name],
    }
    observation["gripper_pose"] = np.array(
        [-1.0, 0.0, 0.0, np.sqrt(2) / 2, 0.0, np.sqrt(2) / 2, 0.0]
    )

    rgb, depth = cam.get_observation()
    _, _, cam_xyza_pts = cam.compute_camera_XYZA(depth)
    homogeneous_pts = np.hstack((cam_xyza_pts, np.ones((cam_xyza_pts.shape[0], 1))))
    transformed_pts = (cam.get_metadata()["mat44"] @ homogeneous_pts.T).T
    world_pts = transformed_pts[:, :3] / transformed_pts[:, 3:]
    valid = np.where(depth < 1)
    observation["points"] = world_pts
    observation["image"] = observation["image"][valid]
    observation["detect_info"]["masks"][0] = target_part_mask[valid]

    run_voxposer(
        my_robot,
        instruction,
        observation,
        category_ids=[target_part_function_name],
    )


def execute_track3_task(
    args,
    sample_dir,
    manual_subdir,
    task_idx,
    gt_steps_raw,
    exec_steps,
    pred_link_function_dict,
    track_dir,
    view,
    img_size,
    flog,
):
    shape_id = manual_subdir.split("_")[0]
    init_pose = load_init_pose(sample_dir)
    env = None
    robot = None

    try:
        env, cam = build_env_and_camera(
            args,
            shape_id,
            init_pose,
            args.obj_init_scale,
            view,
            img_size,
            flog,
        )
        robot = build_robot(env, args.robot_urdf_path)
        link_nameid_dict = {
            name: link_id
            for name, link_id in zip(env.movable_link_names, env.movable_link_ids)
        }
        link_types = load_semantics(shape_id, args.data_dir)
        with open(os.path.join(sample_dir, "all_masks_track3.pkl"), "rb") as file:
            all_masks = pickle.load(file)

        processed_gt_steps = normalize_track3_gt_steps(gt_steps_raw)
        task_dir = ensure_dir(os.path.join(track_dir, f"task_{task_idx}"))
        success_step_count = 0
        success_flag = False

        for step_idx, action_step in enumerate(exec_steps, start=1):
            if step_idx > len(processed_gt_steps):
                break

            gt_step = processed_gt_steps[step_idx - 1]
            if action_step != gt_step:
                break

            target_link_name = gt_steps_raw[step_idx - 1][1].split()[0]
            target_part_function_name, manip_way = action_step.split(": ", 1)
            if target_part_function_name not in pred_link_function_dict:
                flog.write(f"Missing mask alignment for {target_part_function_name}\n")
                break

            target_mask_id = pred_link_function_dict[target_part_function_name]
            target_part_mask_item = get_mask_item(all_masks, target_mask_id)
            if not target_part_mask_item or "segmentation" not in target_part_mask_item:
                flog.write(f"Missing segmentation mask for {target_part_function_name}\n")
                break
            target_part_mask = np.asarray(target_part_mask_item["segmentation"])

            step_save_dir = ensure_dir(os.path.join(task_dir, f"step_{step_idx}"))
            env.set_target_object_part_actor_id(link_nameid_dict[target_link_name])
            init_target_part_qpos = env.get_target_part_qpos()
            link_type = link_types.get(target_link_name, "")
            flog.write(action_step + "\n")

            if "button" in link_type:
                instruction = f"Press {target_part_function_name}, follow rule [{manip_way}]"
                vlm_plan_execution(
                    env,
                    cam,
                    robot,
                    target_part_function_name,
                    target_part_mask,
                    instruction,
                    step_save_dir,
                )
                final_target_part_qpos = env.get_target_part_qpos()
                abs_motion = abs(final_target_part_qpos - init_target_part_qpos)
                joint_id = env.target_object_part_joint_id
                tot_motion = env.joint_angles_upper[joint_id] - env.joint_angles_lower[joint_id] + 1e-8
                success_flag = (abs_motion > 0.01) or (abs_motion / tot_motion > 0.5)
                flog.write(
                    f"{shape_id}_{target_link_name}_{target_part_function_name} "
                    f"abs_motion: {abs_motion} ; tot_motion: {tot_motion} ; success: {success_flag}\n"
                )
            elif "knob" in link_type:
                match = re.search(r"(\d+)", manip_way)
                if not match:
                    success_flag = False
                    break
                rotation_degrees = int(match.group(1))
                instruction = (
                    f"Rotate {target_part_function_name} {rotation_degrees} degrees, "
                    f"follow rule [{manip_way}]"
                )
                vlm_plan_execution(
                    env,
                    cam,
                    robot,
                    target_part_function_name,
                    target_part_mask,
                    instruction,
                    step_save_dir,
                )
                final_target_part_qpos = env.get_target_part_qpos()
                abs_motion = abs(final_target_part_qpos - init_target_part_qpos)
                abs_degrees = round(math.degrees(abs_motion), 2)
                success_flag = rotation_degrees - 30 < abs_degrees < rotation_degrees + 30
                flog.write(
                    f"{shape_id}_{target_link_name}_{target_part_function_name} "
                    f"after rotation abs_motion: {abs_motion} ({abs_degrees}°) ; success: {success_flag}\n"
                )
            elif "door" in link_type:
                instruction = f"{manip_way} {target_part_function_name}"
                vlm_plan_execution(
                    env,
                    cam,
                    robot,
                    target_part_function_name,
                    target_part_mask,
                    instruction,
                    step_save_dir,
                )
                final_target_part_qpos = env.get_target_part_qpos()
                abs_motion = abs(final_target_part_qpos - init_target_part_qpos)
                abs_degrees = round(math.degrees(abs_motion), 2)
                success_flag = abs_degrees > 30
                flog.write(
                    f"{shape_id}_{target_link_name}_{target_part_function_name} "
                    f"door abs_motion: {abs_motion} ({abs_degrees}°) ; success: {success_flag}\n"
                )
            elif any(keyword in link_type for keyword in ["slider", "lid", "screen", "drawer", "container"]):
                mapped_manip = "Pull up" if manip_way == "Open" else "Close"
                instruction = f"{mapped_manip} {target_part_function_name}"
                vlm_plan_execution(
                    env,
                    cam,
                    robot,
                    target_part_function_name,
                    target_part_mask,
                    instruction,
                    step_save_dir,
                )
                final_target_part_qpos = env.get_target_part_qpos()
                abs_motion = abs(final_target_part_qpos - init_target_part_qpos)
                success_flag = abs_motion > 0.01
                flog.write(
                    f"{shape_id}_{target_link_name}_{target_part_function_name} "
                    f"slider-like abs_motion: {abs_motion} ; success: {success_flag}\n"
                )
            else:
                flog.write(f"Unable to execute {action_step}\n")
                success_flag = False

            if not success_flag:
                flog.write(f"Failed at step {step_idx}\n")
                break
            success_step_count += 1

        completion_rate = success_step_count / len(processed_gt_steps) if processed_gt_steps else 0.0
        return success_flag, completion_rate
    finally:
        if robot is not None:
            env.scene.remove_articulation(robot.robot)
        if env is not None:
            env.close()


def benchmark(args, manual_subdir, result_dict, view="front", img_size=1024):
    global total_planning, success_planning, total_execution, success_execution

    shape_id = manual_subdir.split("_")[0]
    sample_dir = os.path.join(args.manual_dir, manual_subdir)
    result_dict[sample_dir] = {
        "total_tasks": [],
        "success_task_plan": [],
        "success_task_execution": [],
        "completion_rates": [],
        "pred_link_function_dict": None,
    }

    print(f"Evaluate {manual_subdir}")

    task_dict, task_json_path = load_task_dict(
        sample_dir,
        global_tasks_path=args.global_tasks_path,
        global_key=args.global_tasks_key,
    )
    manual_pdf_file = find_manual_pdf(sample_dir)
    manual_pngs_dir, _, manual_content = resolve_manual(sample_dir, manual_pdf_file)
    page_vis_info_dict = vis_analyze_page(manual_pngs_dir, vlm_version=args.vlm_version)

    env = None
    flog = open(args.log_path, "a", encoding="utf-8")
    try:
        pred_link_function_dict = align_parts_nocad(
            sample_dir,
            manual_subdir,
            page_vis_info_dict,
            use_cache=not args.no_cache_alignment,
        )
    except RuntimeError:
        init_pose = load_init_pose(sample_dir)
        env, cam = build_env_and_camera(
            args,
            shape_id,
            init_pose,
            args.obj_init_scale,
            view,
            img_size,
            flog,
        )
        pred_link_function_dict = align_parts_nocad(
            sample_dir,
            manual_subdir,
            page_vis_info_dict,
            use_cache=not args.no_cache_alignment,
            env=env,
            cam=cam,
        )

    result_dict[sample_dir]["pred_link_function_dict"] = pred_link_function_dict
    flog.write(f"sample: {manual_subdir}\n")
    flog.write(f"task_json: {task_json_path}\n")
    flog.write(f"pred_link_function_dict: {json.dumps(pred_link_function_dict, ensure_ascii=False)}\n")

    track_dir = ensure_dir(os.path.join(args.out_dir, "track3", manual_subdir))
    task_items = list(task_dict.items())
    if args.task_name:
        task_items = [item for item in task_items if item[0] == args.task_name]
    if args.max_tasks is not None:
        task_items = task_items[: args.max_tasks]

    for task_idx, (task_name, gt_steps) in enumerate(task_items, start=1):
        processed_gt_steps = normalize_track3_gt_steps(gt_steps)
        planned_steps = track3_plan_steps(
            sample_dir,
            task_idx,
            manual_content,
            task_name,
            pred_link_function_dict,
            llm_version=args.llm_version,
            use_cache_plan=not args.no_cache_plan,
        )

        total_planning += 1
        result_dict[sample_dir]["total_tasks"].append(task_name)
        flog.write(f"Task Name: {task_name}\n")
        flog.write(f"GT steps: {processed_gt_steps}\n")
        flog.write(f"planned steps: {planned_steps}\n")

        success_flag = False
        completion_rate = None
        exec_steps = planned_steps
        if args.use_gt_plan:
            exec_steps = list(processed_gt_steps)
            flog.write("Using ground-truth plan for Track 3 execution\n")

        if processed_gt_steps == planned_steps:
            success_planning += 1
            result_dict[sample_dir]["success_task_plan"].append(task_name)
            flog.write(
                f"{shape_id} Correct Planning. Success Planning Rate: "
                f"{success_planning / total_planning}\n"
            )

        if args.execute:
            total_execution += 1
            success_flag, completion_rate = execute_track3_task(
                args,
                sample_dir,
                manual_subdir,
                task_idx,
                gt_steps,
                exec_steps,
                pred_link_function_dict,
                track_dir,
                view,
                img_size,
                flog,
            )
            result_dict[sample_dir]["completion_rates"].append(completion_rate)
            flog.write(f"'{task_name}' Completion Rate: {completion_rate}\n")
            if exec_steps == processed_gt_steps and success_flag:
                success_execution += 1
                result_dict[sample_dir]["success_task_execution"].append(task_name)
                flog.write(
                    f"Successfully complete task '{task_name}'! Success Execution Rate: "
                    f"{success_execution / total_execution}\n"
                )

        with open(
            os.path.join(track_dir, f"task_{task_idx}_summary.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "task_name": task_name,
                    "gt_steps": processed_gt_steps,
                    "planned_steps": planned_steps,
                    "execution_steps": exec_steps,
                    "execution_success": success_flag,
                    "completion_rate": completion_rate,
                },
                file,
                ensure_ascii=False,
                indent=4,
            )

        flog.write("#########################################################\n")

    flog.write("----------------------------------------------------------------\n\n")
    flog.close()
    if env is not None:
        env.close()
    return result_dict


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--manual_dir",
        default=os.path.join(ROOT_DIR, "data", "checkmanual_dataset"),
        type=str,
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(ROOT_DIR, "data", "sapien_dataset"),
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(ROOT_DIR, "results"),
        type=str,
    )
    parser.add_argument("--log_path", default=None, type=str)
    parser.add_argument("--result_path", default=None, type=str)
    parser.add_argument("--global_tasks_path", default=None, type=str)
    parser.add_argument("--global_tasks_key", default=None, type=str)
    parser.add_argument("--sample", default=None, type=str)
    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument("--max_tasks", default=None, type=int)
    parser.add_argument("--obj_init_scale", default=0.4, type=float)
    parser.add_argument("--llm_version", default="gpt-4o", type=str)
    parser.add_argument("--vlm_version", default="gpt-4o", type=str)
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--execute", action="store_true", default=False)
    parser.add_argument("--use_gt_plan", action="store_true", default=False)
    parser.add_argument("--no_cache_alignment", action="store_true", default=False)
    parser.add_argument("--no_cache_plan", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.out_dir = normalize_out_dir(args.out_dir)
    ensure_dir(args.out_dir)
    args.log_path = normalize_results_file(
        args.log_path,
        os.path.join(args.out_dir, "track3.log"),
    )
    args.result_path = normalize_results_file(
        args.result_path,
        os.path.join(args.out_dir, "track3_results.json"),
    )
    args.robot_urdf_path = get_robot_gripper_urdf_path()

    manual_subdirs = list_manual_subdirs(
        args.manual_dir,
        sample=args.sample,
        max_samples=args.max_samples,
    )

    if os.path.exists(args.result_path):
        result_dict = load_json(args.result_path)
    else:
        result_dict = {}

    for manual_subdir in tqdm(manual_subdirs):
        sample_dir = os.path.join(args.manual_dir, manual_subdir)
        if sample_dir in result_dict:
            continue
        result_dict = benchmark(args, manual_subdir, result_dict)
        with open(args.result_path, "w", encoding="utf-8") as file:
            json.dump(result_dict, file, ensure_ascii=False, indent=4)

    if total_planning > 0:
        print(f"Track 3 planning SR: {success_planning / total_planning:.4f}")
    if total_execution > 0:
        print(f"Track 3 execution SR: {success_execution / total_execution:.4f}")
