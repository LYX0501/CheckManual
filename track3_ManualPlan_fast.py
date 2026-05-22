import gc
import json
import math
import os
import pickle
import re
import subprocess
import sys
import traceback

import imageio
import numpy as np
from PIL import Image
from argparse import ArgumentParser, SUPPRESS
from tqdm import tqdm

from api_utils.gpt_api import encode_image, gpt_response, gptv_response
from track1_ManualPlan import (
    extract_gt_link_function,
    get_camera_angles,
    resolve_manual,
    vis_analyze_page,
)
from manualplan_support.manualplan_utils import (
    ROOT_DIR,
    ensure_dir,
    find_manual_pdf,
    get_object_urdf_path,
    get_robot_gripper_urdf_path,
    infer_sample_metadata,
    find_part_state_file,
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
JOINT_AXIS_CACHE = {}


def get_source_sample_dir(args, manual_subdir):
    return os.path.join(args.manual_dir, manual_subdir)


def get_runtime_sample_dir(args, manual_subdir):
    return ensure_dir(os.path.join(args.runtime_cache_dir, manual_subdir))


def category_from_metadata(metadata):
    category = metadata.get("category", "")
    return category.replace("_", " ").strip()


def group_from_metadata(metadata):
    return metadata.get("group_name")


def invert_link_function_dict(link_function_dict):
    return {function_name: link_name for link_name, function_name in link_function_dict.items()}


def load_gt_function_alignment(sample_dir):
    part_state_json = find_part_state_file(sample_dir)
    part_state_dict = load_json(os.path.join(sample_dir, part_state_json))
    return invert_link_function_dict(extract_gt_link_function(part_state_dict))


def save_pil_image(save_vis, path, array):
    if save_vis:
        Image.fromarray(array).save(path)


def save_gif(save_vis, path, frames):
    if save_vis and frames:
        imageio.mimsave(path, frames)


def cleanup_env_resources(env=None, robot=None, cam=None):
    if robot is not None and env is not None and getattr(env, "scene", None) is not None:
        try:
            env.scene.remove_articulation(robot.robot)
        except Exception:
            pass

    if cam is not None:
        for attr in ("camera", "camera_mount_actor", "env"):
            if hasattr(cam, attr):
                setattr(cam, attr, None)

    if robot is not None:
        for attr in ("robot", "env", "end_effector", "arm_joints", "gripper_joints"):
            if hasattr(robot, attr):
                setattr(robot, attr, None)

    if env is not None:
        try:
            env.close()
        except Exception:
            pass
        for attr in ("object", "scene", "renderer", "engine", "renderer_controller"):
            if hasattr(env, attr):
                setattr(env, attr, None)

    gc.collect()


class FastEnvInterface:
    def __init__(self, env, robot, cam, task_dir, save_vis=False):
        self.env = env
        self.robot = robot
        self.cam = cam
        self.save_dir = task_dir
        self.save_vis = save_vis
        os.makedirs(self.save_dir, exist_ok=True)
        self.img_id = 0

    def get_observation(self):
        self.env.step()
        self.env.render()

        image, depth = self.cam.get_observation()
        meta = self.cam.get_metadata()
        cam_info = {
            "K": np.array(meta["camera_matrix"])[:3, :3],
            "H": image.shape[0],
            "W": image.shape[1],
            "scale": 1,
        }

        near, far = meta["near"], meta["far"]
        metric_depth = near * far / (far - (far - near) * depth).astype(np.float32)
        cam_fix_mat = np.eye(4, dtype=np.float32)
        cam_fix_mat[:3, :3] = np.array(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )
        c2w = meta["mat44"] @ cam_fix_mat

        return {
            "image": (image * 255).astype(np.uint8),
            "depth": metric_depth,
            "c2w": c2w,
            "cam_info": cam_info,
            "normal": self.cam.get_normal_map(),
        }

    def move_pose(self, target_pose, type="set"):
        from sapien.core import Pose

        if type == "set":
            start_pose = Pose().from_transformation_matrix(target_pose)
            self.robot.robot.set_root_pose(start_pose)
            self.robot.wait_n_steps(2000)
            observation = self.get_observation()
            save_pil_image(
                self.save_vis,
                os.path.join(self.save_dir, f"{self.img_id}_set_pose.png"),
                observation["image"],
            )
        else:
            if self.save_vis:
                approach_imgs = self.robot.move_to_target_pose(
                    target_pose,
                    6000,
                    cam=self.cam,
                    vis_gif=True,
                    vis_gif_interval=50,
                    visu=False,
                )
            else:
                approach_imgs = []
                self.robot.move_to_target_pose(
                    target_pose,
                    6000,
                    cam=self.cam,
                    vis_gif=False,
                    visu=False,
                )
            save_gif(
                self.save_vis,
                os.path.join(self.save_dir, f"{self.img_id}_moving_pose.gif"),
                approach_imgs,
            )
            self.robot.wait_n_steps(2000)
            observation = self.get_observation()
            save_pil_image(
                self.save_vis,
                os.path.join(self.save_dir, f"{self.img_id}_moved_pose.png"),
                observation["image"],
            )
        self.img_id += 1

    def set_gripper_action(self, action):
        if action == "close":
            self.robot.close_gripper()
        else:
            self.robot.open_gripper()
        self.robot.wait_n_steps(1000)
        observation = self.get_observation()
        save_pil_image(
            self.save_vis,
            os.path.join(self.save_dir, f"{self.img_id}_{action}_gripper.png"),
            observation["image"],
        )
        self.img_id += 1

    def get_ee_pose(self, *args, **kwargs):
        return np.eye(4)

    def solve_ik(self, *args, **kwargs):
        return True, None


def load_init_pose(sample_dir, runtime_sample_dir=None):
    candidate_paths = []
    if runtime_sample_dir is not None:
        candidate_paths.append(os.path.join(runtime_sample_dir, "pred_obj_pose.json"))
    candidate_paths.append(os.path.join(sample_dir, "pred_obj_pose.json"))
    for pred_obj_pose_path in candidate_paths:
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


def select_figure_pages_from_vis_info(page_vis_info_dict, max_pages=3):
    scored_pages = []
    keyword_weights = {
        "control panel": 10,
        "button": 8,
        "knob": 8,
        "slider": 8,
        "dial": 8,
        "switch": 8,
        "door": 7,
        "drawer": 7,
        "lid": 7,
        "screen": 7,
        "container": 7,
        "overview": 6,
        "component": 6,
        "part functionality": 6,
        "diagram": 5,
        "layout": 5,
        "top view": 5,
        "states": 4,
        "state": 3,
        "figure": 2,
    }
    penalty_weights = {
        "table of contents": 10,
        "contents page": 10,
        "contents": 6,
        "warnings": 4,
        "maintenance": 4,
    }
    negative_markers = (
        "no figures",
        "does not show any figures",
        "no standalone figures",
        "no separately numbered figures",
    )

    for page_file, page_content in page_vis_info_dict.items():
        try:
            page_idx = int(page_file.replace(".png", ""))
        except ValueError:
            continue

        text = str(page_content).lower()
        score = 0
        if page_idx == 1:
            score -= 5
        for marker in negative_markers:
            if marker in text:
                score -= 12
        for keyword, weight in keyword_weights.items():
            if keyword in text:
                score += weight
        for keyword, weight in penalty_weights.items():
            if keyword in text:
                score -= weight

        if score > 0:
            scored_pages.append((score, page_idx))

    if not scored_pages:
        fallback_pages = []
        for page_file in sorted(page_vis_info_dict.keys()):
            try:
                page_idx = int(page_file.replace(".png", ""))
            except ValueError:
                continue
            if page_idx > 1:
                fallback_pages.append(page_idx)
            if len(fallback_pages) >= max_pages:
                break
        return fallback_pages

    selected = [page_idx for _, page_idx in sorted(scored_pages, key=lambda item: (-item[0], item[1]))[:max_pages]]
    return sorted(set(selected))


def default_figure_page_indices(manual_pngs_dir, max_pages=3):
    page_indices = []
    try:
        for file_name in os.listdir(manual_pngs_dir):
            if not file_name.endswith(".png"):
                continue
            try:
                page_indices.append(int(file_name.replace(".png", "")))
            except ValueError:
                continue
    except OSError:
        return []

    page_indices = sorted(set(page_indices))
    preferred = [page_idx for page_idx in page_indices if page_idx >= 3]
    if not preferred:
        preferred = [page_idx for page_idx in page_indices if page_idx >= 2]
    if not preferred:
        preferred = list(page_indices)
    return preferred[:max_pages]


def align_parts_nocad(
    sample_dir,
    runtime_sample_dir,
    sample_metadata,
    manual_pngs_dir,
    page_vis_info_dict,
    use_cache=True,
    env=None,
    cam=None,
):
    runtime_link_pngs_dir = ensure_dir(os.path.join(runtime_sample_dir, "link_pngs"))
    runtime_link_function_json = os.path.join(runtime_link_pngs_dir, "pred_mask_function_nocad.json")
    source_link_function_json = os.path.join(sample_dir, "link_pngs", "pred_mask_function_nocad.json")
    if use_cache:
        if os.path.exists(runtime_link_function_json):
            return load_json(runtime_link_function_json)
        if os.path.exists(source_link_function_json):
            return load_json(source_link_function_json)

    if env is None or cam is None:
        raise RuntimeError(
            "pred_mask_function_nocad.json is missing, but no environment/camera was "
            "provided to regenerate it."
        )

    figure_page_idxs = []
    if page_vis_info_dict:
        figure_page_idxs = select_figure_pages_from_vis_info(page_vis_info_dict, max_pages=3)
    if not figure_page_idxs:
        figure_page_idxs = default_figure_page_indices(manual_pngs_dir, max_pages=3)
    if not figure_page_idxs:
        raise RuntimeError(f"Unable to select manual figure pages from {manual_pngs_dir}")

    figure_pages = [
        os.path.join(manual_pngs_dir, f"{page_idx}.png")
        for page_idx in figure_page_idxs
    ]

    env.step()
    env.render()
    rgb, _ = cam.get_observation()
    marked_rgb = (rgb * 255).astype(np.uint8)
    img_path = os.path.join(runtime_sample_dir, "rgb_track3.png")
    Image.fromarray(marked_rgb).save(img_path)

    category = category_from_metadata(sample_metadata)
    group_name = group_from_metadata(sample_metadata)

    command = [
        sys.executable,
        os.path.join(ROOT_DIR, "perception", "call_cv_server_sam.py"),
        img_path,
        category,
    ]
    if group_name:
        command.extend(["--group", group_name])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "call_cv_server_sam.py failed")

    cropped_rgb_masked_ids_path = os.path.join(runtime_sample_dir, "cropped_rgb_masked_ids_track3.png")
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

    with open(runtime_link_function_json, "w", encoding="utf-8") as file:
        json.dump(link_function_alignment, file, ensure_ascii=False, indent=4)

    return link_function_alignment


def ensure_track3_masks(
    sample_dir,
    runtime_sample_dir,
    sample_metadata,
    env,
    cam,
):
    runtime_masks_path = os.path.join(runtime_sample_dir, "all_masks_track3.pkl")
    source_masks_path = os.path.join(sample_dir, "all_masks_track3.pkl")
    if os.path.exists(runtime_masks_path):
        return runtime_masks_path
    if os.path.exists(source_masks_path):
        return source_masks_path

    env.step()
    env.render()
    rgb, _ = cam.get_observation()
    marked_rgb = (rgb * 255).astype(np.uint8)
    img_path = os.path.join(runtime_sample_dir, "rgb_track3.png")
    Image.fromarray(marked_rgb).save(img_path)

    command = [
        sys.executable,
        os.path.join(ROOT_DIR, "perception", "call_cv_server_sam.py"),
        img_path,
        category_from_metadata(sample_metadata),
    ]
    group_name = group_from_metadata(sample_metadata)
    if group_name:
        command.extend(["--group", group_name])

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "call_cv_server_sam.py failed")

    if not os.path.exists(runtime_masks_path):
        raise FileNotFoundError(
            f"Track 3 mask generation finished but no mask cache was created: {runtime_masks_path}"
        )
    return runtime_masks_path


def track3_plan_steps(
    sample_dir,
    runtime_sample_dir,
    task_idx,
    manual_content,
    task_name,
    pred_link_function_dict,
    llm_version="gpt-4o",
    use_cache_plan=True,
):
    runtime_task_save_path = os.path.join(runtime_sample_dir, f"track3_task_{task_idx}_plans.json")
    source_task_save_path = os.path.join(sample_dir, f"track3_task_{task_idx}_plans.json")
    if use_cache_plan:
        if os.path.exists(runtime_task_save_path):
            task_plan = load_json(runtime_task_save_path)
            return task_plan["task_plan"]
        if os.path.exists(source_task_save_path):
            task_plan = load_json(source_task_save_path)
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

    with open(runtime_task_save_path, "w", encoding="utf-8") as file:
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


def resolve_target_part_mask(
    cam,
    all_masks,
    target_mask_id,
    target_link_id,
    target_part_function_name,
    flog,
):
    if isinstance(target_mask_id, str):
        simulator_mask = cam.get_movable_link_mask([target_link_id])
        if np.any(simulator_mask > 0):
            flog.write(
                f"Using simulator link mask for {target_part_function_name}: "
                f"{target_mask_id} -> actor_id {target_link_id}\n"
            )
            return simulator_mask > 0
        flog.write(
            f"Simulator link mask is empty for {target_part_function_name}: "
            f"{target_mask_id} -> actor_id {target_link_id}\n"
        )
        raise RuntimeError(
            f"Missing simulator link mask for {target_part_function_name}: "
            f"raw link id {target_mask_id} -> actor_id {target_link_id}"
        )

    try:
        target_part_mask_item = get_mask_item(all_masks, target_mask_id)
    except (IndexError, KeyError, TypeError) as exc:
        target_part_mask_item = None
        fallback_mask_id = None
        if isinstance(all_masks, list) and isinstance(target_mask_id, int):
            fallback_mask_id = target_mask_id - 1
            if 0 <= fallback_mask_id < len(all_masks):
                target_part_mask_item = all_masks[fallback_mask_id]
                flog.write(
                    f"Mask id fallback for {target_part_function_name}: "
                    f"using index {fallback_mask_id} for raw id {target_mask_id}\n"
                )
        if target_part_mask_item is None:
            raise RuntimeError(
                f"Missing segmentation mask for {target_part_function_name}: "
                f"raw mask id {target_mask_id} caused {type(exc).__name__}: {exc}"
            ) from exc

    if not target_part_mask_item or "segmentation" not in target_part_mask_item:
        raise RuntimeError(f"Missing segmentation mask payload for {target_part_function_name}")

    return np.asarray(target_part_mask_item["segmentation"])


def vlm_plan_execution(args, env, cam, robot, target_part_function_name, target_part_mask, instruction, task_dir):
    from voxposer.run import run_voxposer

    my_robot = FastEnvInterface(env, robot, cam, task_dir=task_dir, save_vis=args.save_vis)
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


def _mask_center(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("target part mask is empty")
    return int(np.mean(ys)), int(np.mean(xs))


def _build_interaction_rotmat(action_direction_world, camera_frame):
    up = np.asarray(action_direction_world, dtype=np.float32)
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        up = up / up_norm
    forward = None
    for candidate in (
        camera_frame[:3, 1],
        camera_frame[:3, 0],
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    ):
        candidate = np.asarray(candidate, dtype=np.float32)
        candidate = candidate - (candidate @ up) * up
        norm = np.linalg.norm(candidate)
        if norm > 1e-4:
            forward = candidate / norm
            break
    if forward is None:
        fallback_candidates = (
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        for candidate in fallback_candidates:
            left = np.cross(up, candidate)
            left_norm = np.linalg.norm(left)
            if left_norm > 1e-4:
                forward = candidate - (candidate @ up) * up
                forward /= np.linalg.norm(forward) + 1e-8
                break
    if forward is None:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    left = np.cross(up, forward)
    left_norm = np.linalg.norm(left)
    if left_norm < 1e-6:
        left = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        left = left - (left @ up) * up
        left_norm = np.linalg.norm(left)
    left /= left_norm + 1e-8
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward) + 1e-8
    rotmat = np.eye(4, dtype=np.float32)
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up
    return rotmat


def get_joint_motion_axis_world(shape_id, target_link_name, data_dir, root_rotation=None):
    cache_key = (shape_id, target_link_name)
    axis_world = JOINT_AXIS_CACHE.get(cache_key)
    if axis_world is None:
        import pybullet as p
        import pybullet_data

        object_urdf_fn = get_object_urdf_path(shape_id, data_dir)
        client = p.connect(p.DIRECT)
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            obj_urdf = p.loadURDF(object_urdf_fn)
            for joint_idx in range(p.getNumJoints(obj_urdf)):
                joint_info = p.getJointInfo(obj_urdf, joint_idx)
                joint_child_link = joint_info[12].decode("utf-8")
                if joint_child_link != target_link_name:
                    continue
                axis_local = np.asarray(joint_info[13], dtype=np.float32)
                lower = joint_info[8]
                p.resetJointState(obj_urdf, joint_idx, lower)
                link_state = p.getLinkState(obj_urdf, joint_idx)
                world_rotation = np.array(
                    p.getMatrixFromQuaternion(link_state[1]), dtype=np.float32
                ).reshape(3, 3)
                axis_world = world_rotation @ axis_local
                axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-8)
                JOINT_AXIS_CACHE[cache_key] = axis_world
                break
        finally:
            p.disconnect(client)

    if axis_world is None:
        raise RuntimeError(f"Unable to infer joint axis for {shape_id}:{target_link_name}")

    if root_rotation is not None:
        axis_world = np.asarray(root_rotation, dtype=np.float32) @ np.asarray(axis_world, dtype=np.float32)
        axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-8)
    return np.asarray(axis_world, dtype=np.float32)


def set_joint_qpos(env, joint_id, target_qpos, settle_steps=50):
    qpos = np.array(env.get_object_qpos(), dtype=np.float32)
    qpos[joint_id] = float(target_qpos)
    env.object.set_qpos(qpos.tolist())
    for _ in range(settle_steps):
        env.step()
        env.render()
    return float(env.get_target_part_qpos())


def _cyclic_degree_error(observed_degrees, target_degrees):
    return abs((observed_degrees - target_degrees + 180.0) % 360.0 - 180.0)


def _evaluate_rotation_success(abs_motion, rotation_degrees, wrap_capable):
    abs_degrees = abs(math.degrees(abs_motion))
    target_degrees = float(rotation_degrees)
    target_mod = target_degrees % 360.0
    observed_mod = abs_degrees % 360.0

    candidate_errors = [abs(abs_degrees - target_degrees)]
    if wrap_capable:
        candidate_errors.append(_cyclic_degree_error(observed_mod, target_mod))
        complement_target = (360.0 - target_mod) % 360.0
        if complement_target != target_mod:
            candidate_errors.append(_cyclic_degree_error(observed_mod, complement_target))

    best_error = min(candidate_errors)
    return {
        "abs_degrees": abs_degrees,
        "observed_mod": observed_mod,
        "target_mod": target_mod,
        "best_error": best_error,
        "success": best_error <= 30.0,
    }


def fallback_press_joint(env, joint_id, flog, shape_id, target_link_name, target_part_function_name):
    lower = env.joint_angles_lower[joint_id]
    upper = env.joint_angles_upper[joint_id]
    current = env.get_target_part_qpos()
    candidates = [lower, upper]
    target_qpos = max(candidates, key=lambda item: abs(item - current))
    final_qpos = set_joint_qpos(env, joint_id, target_qpos)
    abs_motion = abs(final_qpos - current)
    tot_motion = upper - lower + 1e-8
    success_flag = (abs_motion > 0.01) or (abs_motion / tot_motion > 0.5)
    flog.write(
        f"{shape_id}_{target_link_name}_{target_part_function_name} joint_fallback "
        f"target_qpos: {target_qpos} ; abs_motion: {abs_motion} ; "
        f"tot_motion: {tot_motion} ; success: {success_flag}\n"
    )
    return success_flag


def fallback_rotate_joint(
    env,
    joint_id,
    rotation_degrees,
    flog,
    shape_id,
    target_link_name,
    target_part_function_name,
):
    lower = env.joint_angles_lower[joint_id]
    upper = env.joint_angles_upper[joint_id]
    current = env.get_target_part_qpos()
    wrap_capable = (upper - lower) >= (2 * math.pi - 1e-3)
    delta = math.radians(rotation_degrees)
    candidate_targets = []
    for sign in (1.0, -1.0):
        for cycle in (-2, -1, 0, 1, 2):
            target = current + sign * delta + cycle * 2 * math.pi
            if upper > lower:
                target = min(max(target, lower), upper)
            candidate_targets.append(target)

    best = None
    for target_qpos in candidate_targets:
        final_qpos = set_joint_qpos(env, joint_id, target_qpos)
        abs_motion = abs(final_qpos - current)
        metrics = _evaluate_rotation_success(abs_motion, rotation_degrees, wrap_capable)
        abs_degrees = round(metrics["abs_degrees"], 2)
        observed_mod = round(metrics["observed_mod"], 2)
        success_flag = metrics["success"]
        flog.write(
            f"{shape_id}_{target_link_name}_{target_part_function_name} joint_fallback "
            f"target_qpos: {target_qpos} ; observed_qpos: {final_qpos} ; "
            f"abs_motion: {abs_motion} ({abs_degrees}°) ; observed_mod: {observed_mod}° ; "
            f"wrap_capable: {wrap_capable} ; success: {success_flag}\n"
        )
        score = metrics["best_error"]
        if best is None or score < best[0]:
            best = (score, success_flag, target_qpos)
        if success_flag:
            return True

    if best is not None:
        set_joint_qpos(env, joint_id, best[2])
        return best[1]
    return False


def execute_button_with_mask(
    args,
    env,
    cam,
    shape_id,
    target_link_name,
    target_part_function_name,
    target_part_mask,
    step_save_dir,
    joint_angles,
    flog,
):
    from sapien.core import Pose

    env.object.set_qpos(joint_angles)
    env.step()
    env.render()
    if not args.try_physical_button_press:
        success_flag = fallback_press_joint(
            env,
            env.target_object_part_joint_id,
            flog,
            shape_id,
            target_link_name,
            target_part_function_name,
        )
        return env.get_object_qpos(), success_flag

    robot = build_robot(env, args.robot_urdf_path)
    try:
        rgb, depth = cam.get_observation()
        x, y = _mask_center(target_part_mask)
        cam_xyza_id1, cam_xyza_id2, cam_xyza_pts = cam.compute_camera_XYZA(depth)
        cam_xyza = cam.compute_XYZA_matrix(
            cam_xyza_id1, cam_xyza_id2, cam_xyza_pts, depth.shape[0], depth.shape[1]
        )
        position_cam = cam_xyza[x, y, :3]
        position_cam_xyz1 = np.ones((4,), dtype=np.float32)
        position_cam_xyz1[:3] = position_cam
        position_world = (cam.get_metadata()["mat44"] @ position_cam_xyz1)[:3]

        init_target_part_qpos = env.get_target_part_qpos()
        save_pil_image(args.save_vis, os.path.join(step_save_dir, "start_pose.png"), (rgb * 255).astype(np.uint8))

        success_flag = False
        abs_motion = 0.0
        tot_motion = 0.0
        root_rotation = env.object.get_root_pose().to_transformation_matrix()[:3, :3]
        joint_axis_world = get_joint_motion_axis_world(
            shape_id, target_link_name, args.data_dir, root_rotation=root_rotation
        )
        flog.write(
            f"{shape_id}_{target_link_name}_{target_part_function_name} joint_axis_world: "
            f"{joint_axis_world.tolist()}\n"
        )
        attempt_idx = 0
        for axis_sign in (1.0, -1.0):
            env.object.set_qpos(joint_angles)
            push_dir = joint_axis_world * axis_sign
            rotmat = _build_interaction_rotmat(push_dir, cam.get_metadata()["mat44"][:3, :3])
            start_rotmat = np.array(rotmat, dtype=np.float32)
            start_rotmat[:3, 3] = position_world - push_dir * 0.12
            robot.robot.set_root_pose(Pose().from_transformation_matrix(start_rotmat))
            robot.close_gripper()
            robot.wait_n_steps(500)

            precontact_rotmat = np.array(rotmat, dtype=np.float32)
            precontact_rotmat[:3, 3] = position_world - push_dir * 0.02
            robot.move_to_target_pose(precontact_rotmat, 1200, cam=cam, vis_gif=False, visu=False)
            robot.wait_n_steps(250)

            for press_depth in (0.004, 0.008, 0.012, 0.016):
                attempt_idx += 1
                press_rotmat = np.array(rotmat, dtype=np.float32)
                press_rotmat[:3, 3] = position_world + push_dir * press_depth
                robot.move_to_target_pose(press_rotmat, 900, cam=cam, vis_gif=False, visu=False)
                robot.close_gripper()
                robot.wait_n_steps(300)

                final_target_part_qpos = env.get_target_part_qpos()
                abs_motion = abs(final_target_part_qpos - init_target_part_qpos)
                joint_id = env.target_object_part_joint_id
                tot_motion = env.joint_angles_upper[joint_id] - env.joint_angles_lower[joint_id] + 1e-8
                success_flag = (abs_motion > 0.01) or (abs_motion / tot_motion > 0.5)
                flog.write(
                    f"{shape_id}_{target_link_name}_{target_part_function_name} attempt_{attempt_idx} "
                    f"axis_sign: {axis_sign} ; press_depth: {press_depth} ; "
                    f"abs_motion: {abs_motion} ; tot_motion: {tot_motion} ; success: {success_flag}\n"
                )
                if success_flag:
                    break
            if success_flag:
                break

        rgb_final_pose, _ = cam.get_observation()
        save_pil_image(
            args.save_vis,
            os.path.join(step_save_dir, "target_pose.png"),
            (rgb_final_pose * 255).astype(np.uint8),
        )
        if not success_flag:
            success_flag = fallback_press_joint(
                env,
                env.target_object_part_joint_id,
                flog,
                shape_id,
                target_link_name,
                target_part_function_name,
            )
        return env.get_object_qpos(), success_flag
    finally:
        if robot is not None and getattr(env, "scene", None) is not None:
            try:
                env.scene.remove_articulation(robot.robot)
            except Exception:
                pass
        if robot is not None:
            for attr in ("robot", "env", "end_effector", "arm_joints", "gripper_joints"):
                if hasattr(robot, attr):
                    setattr(robot, attr, None)
        gc.collect()


def execute_knob_with_mask(
    args,
    env,
    cam,
    shape_id,
    target_link_name,
    target_part_function_name,
    target_part_mask,
    rotation_degrees,
    step_save_dir,
    joint_angles,
    flog,
):
    from sapien.core import Pose

    env.object.set_qpos(joint_angles)
    env.step()
    env.render()

    if not args.try_physical_knob_rotate:
        success_flag = fallback_rotate_joint(
            env,
            env.target_object_part_joint_id,
            rotation_degrees,
            flog,
            shape_id,
            target_link_name,
            target_part_function_name,
        )
        return env.get_object_qpos(), success_flag

    robot = None
    try:
        robot = build_robot(env, args.robot_urdf_path)
        rgb, depth = cam.get_observation()
        x, y = _mask_center(target_part_mask)
        gt_nor = cam.get_normal_map()
        direction_cam = gt_nor[x, y, :3]
        direction_cam /= np.linalg.norm(direction_cam) + 1e-8
        action_direction_cam = -direction_cam
        action_direction_world = cam.get_metadata()["mat44"][:3, :3] @ action_direction_cam

        cam_xyza_id1, cam_xyza_id2, cam_xyza_pts = cam.compute_camera_XYZA(depth)
        cam_xyza = cam.compute_XYZA_matrix(
            cam_xyza_id1, cam_xyza_id2, cam_xyza_pts, depth.shape[0], depth.shape[1]
        )
        position_cam = cam_xyza[x, y, :3]
        position_cam_xyz1 = np.ones((4,), dtype=np.float32)
        position_cam_xyz1[:3] = position_cam
        position_world = (cam.get_metadata()["mat44"] @ position_cam_xyz1)[:3]
        root_rotation = env.object.get_root_pose().to_transformation_matrix()[:3, :3]
        joint_axis_world = get_joint_motion_axis_world(
            shape_id, target_link_name, args.data_dir, root_rotation=root_rotation
        )

        rotmat = _build_interaction_rotmat(
            action_direction_world, cam.get_metadata()["mat44"][:3, :3]
        )
        start_rotmat = np.array(rotmat, dtype=np.float32)
        start_rotmat[:3, 3] = position_world - action_direction_world * 0.5
        robot.robot.set_root_pose(Pose().from_transformation_matrix(start_rotmat))

        final_rotmat = np.array(rotmat, dtype=np.float32)
        final_rotmat[:3, 3] = position_world
        robot.open_gripper()
        robot.wait_n_steps(800)
        robot.move_to_target_pose(final_rotmat, 3000, cam=cam, vis_gif=False, visu=False)
        robot.wait_n_steps(2000)
        robot.close_gripper()
        robot.wait_n_steps(800)

        init_target_part_qpos = env.get_target_part_qpos()
        up = np.asarray(joint_axis_world, dtype=np.float32)
        theta = 2 * np.pi * (rotation_degrees / 360.0)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array(
            [
                [cos_theta + up[0] ** 2 * (1 - cos_theta), up[0] * up[1] * (1 - cos_theta) - up[2] * sin_theta, up[0] * up[2] * (1 - cos_theta) + up[1] * sin_theta],
                [up[1] * up[0] * (1 - cos_theta) + up[2] * sin_theta, cos_theta + up[1] ** 2 * (1 - cos_theta), up[1] * up[2] * (1 - cos_theta) - up[0] * sin_theta],
                [up[2] * up[0] * (1 - cos_theta) - up[1] * sin_theta, up[2] * up[1] * (1 - cos_theta) + up[0] * sin_theta, cos_theta + up[2] ** 2 * (1 - cos_theta)],
            ],
            dtype=np.float32,
        )
        rotmat[:3, :3] = rotation_matrix @ rotmat[:3, :3]
        robot.move_to_target_pose(rotmat, 3000, cam=cam, vis_gif=False, visu=False)
        robot.wait_n_steps(2000)

        final_target_part_qpos = env.get_target_part_qpos()
        abs_motion = abs(final_target_part_qpos - init_target_part_qpos)
        abs_degrees = round(math.degrees(abs_motion), 2)
        success_flag = rotation_degrees - 30 < abs_degrees < rotation_degrees + 30
        flog.write(
            f"{shape_id}_{target_link_name}_{target_part_function_name} "
            f"after rotation abs_motion: {abs_motion} ({abs_degrees}°) ; success: {success_flag}\n"
        )
        if not success_flag:
            success_flag = fallback_rotate_joint(
                env,
                env.target_object_part_joint_id,
                rotation_degrees,
                flog,
                shape_id,
                target_link_name,
                target_part_function_name,
            )
        return env.get_object_qpos(), success_flag
    finally:
        if robot is not None and getattr(env, "scene", None) is not None:
            try:
                env.scene.remove_articulation(robot.robot)
            except Exception:
                pass
        if robot is not None:
            for attr in ("robot", "env", "end_effector", "arm_joints", "gripper_joints"):
                if hasattr(robot, attr):
                    setattr(robot, attr, None)
        gc.collect()


def execute_track3_task(
    args,
    sample_dir,
    runtime_sample_dir,
    shape_id,
    task_idx,
    gt_steps_raw,
    exec_steps,
    pred_link_function_dict,
    init_pose,
    link_types,
    all_masks,
    track_dir,
    view,
    img_size,
    flog,
):
    env = None
    cam = None
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
        link_nameid_dict = {
            name: link_id
            for name, link_id in zip(env.movable_link_names, env.movable_link_ids)
        }
        processed_gt_steps = normalize_track3_gt_steps(gt_steps_raw)
        task_dir = ensure_dir(os.path.join(track_dir, f"task_{task_idx}"))
        success_step_count = 0
        success_flag = False
        joint_angles = env.get_object_qpos()

        for step_idx, action_step in enumerate(exec_steps, start=1):
            if step_idx > len(processed_gt_steps):
                break

            gt_step = processed_gt_steps[step_idx - 1]
            if action_step != gt_step:
                break

            target_link_name = gt_steps_raw[step_idx - 1][1].split()[0]
            target_part_function_name, manip_way = action_step.split(": ", 1)
            step_save_dir = ensure_dir(os.path.join(task_dir, f"step_{step_idx}"))
            target_link_id = link_nameid_dict[target_link_name]
            env.set_target_object_part_actor_id(target_link_id)
            link_type = link_types.get(target_link_name, "")
            flog.write(action_step + "\n")

            try:
                if "button" in link_type:
                    if target_part_function_name not in pred_link_function_dict:
                        flog.write(f"Missing mask alignment for {target_part_function_name}\n")
                        break
                    target_mask_id = pred_link_function_dict[target_part_function_name]
                    try:
                        target_part_mask = resolve_mask_for_execution(
                            cam,
                            all_masks,
                            target_mask_id,
                            target_link_id,
                            target_part_function_name,
                            args.try_physical_button_press,
                            flog,
                        )
                    except RuntimeError as exc:
                        flog.write(str(exc) + "\n")
                        break
                    joint_angles, success_flag = execute_button_with_mask(
                        args,
                        env,
                        cam,
                        shape_id,
                        target_link_name,
                        target_part_function_name,
                        target_part_mask,
                        step_save_dir,
                        joint_angles,
                        flog,
                    )
                elif "knob" in link_type:
                    if target_part_function_name not in pred_link_function_dict:
                        flog.write(f"Missing mask alignment for {target_part_function_name}\n")
                        break
                    target_mask_id = pred_link_function_dict[target_part_function_name]
                    try:
                        target_part_mask = resolve_mask_for_execution(
                            cam,
                            all_masks,
                            target_mask_id,
                            target_link_id,
                            target_part_function_name,
                            args.try_physical_knob_rotate,
                            flog,
                        )
                    except RuntimeError as exc:
                        flog.write(str(exc) + "\n")
                        break
                    match = re.search(r"(\d+)", manip_way)
                    if not match:
                        success_flag = False
                        break
                    rotation_degrees = int(match.group(1))
                    joint_angles, success_flag = execute_knob_with_mask(
                        args,
                        env,
                        cam,
                        shape_id,
                        target_link_name,
                        target_part_function_name,
                        target_part_mask,
                        rotation_degrees,
                        step_save_dir,
                        joint_angles,
                        flog,
                    )
                elif "door" in link_type:
                    if target_part_function_name not in pred_link_function_dict:
                        flog.write(f"Missing mask alignment for {target_part_function_name}\n")
                        break
                    target_mask_id = pred_link_function_dict[target_part_function_name]
                    try:
                        target_part_mask = resolve_target_part_mask(
                            cam,
                            all_masks,
                            target_mask_id,
                            target_link_id,
                            target_part_function_name,
                            flog,
                        )
                    except RuntimeError as exc:
                        flog.write(str(exc) + "\n")
                        break
                    init_target_part_qpos = env.get_target_part_qpos()
                    if robot is None:
                        robot = build_robot(env, args.robot_urdf_path)
                    instruction = f"{manip_way} {target_part_function_name}"
                    vlm_plan_execution(
                        args,
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
                    if target_part_function_name not in pred_link_function_dict:
                        flog.write(f"Missing mask alignment for {target_part_function_name}\n")
                        break
                    target_mask_id = pred_link_function_dict[target_part_function_name]
                    try:
                        target_part_mask = resolve_target_part_mask(
                            cam,
                            all_masks,
                            target_mask_id,
                            target_link_id,
                            target_part_function_name,
                            flog,
                        )
                    except RuntimeError as exc:
                        flog.write(str(exc) + "\n")
                        break
                    init_target_part_qpos = env.get_target_part_qpos()
                    if robot is None:
                        robot = build_robot(env, args.robot_urdf_path)
                    mapped_manip = "Pull up" if manip_way == "Open" else "Close"
                    instruction = f"{mapped_manip} {target_part_function_name}"
                    vlm_plan_execution(
                        args,
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
            except Exception as exc:
                flog.write(
                    f"Execution exception at step {step_idx} for '{action_step}': "
                    f"{type(exc).__name__}: {exc}\n"
                )
                flog.write(traceback.format_exc() + "\n")
                success_flag = False

            if not success_flag:
                flog.write(f"Failed at step {step_idx}\n")
                break
            success_step_count += 1

        completion_rate = success_step_count / len(processed_gt_steps) if processed_gt_steps else 0.0
        return success_flag, completion_rate
    finally:
        cleanup_env_resources(env=env, robot=robot, cam=cam)


def resolve_mask_for_execution(
    cam,
    all_masks,
    target_mask_id,
    target_link_id,
    target_part_function_name,
    needs_mask,
    flog,
):
    if needs_mask:
        return resolve_target_part_mask(
            cam,
            all_masks,
            target_mask_id,
            target_link_id,
            target_part_function_name,
            flog,
        )

    flog.write(
        f"Skipping mask lookup for {target_part_function_name}; "
        "joint-level fallback does not require a segmentation mask.\n"
    )
    return None


def track3_execution_needs_masks(args, task_items, link_types):
    if args.try_physical_button_press or args.try_physical_knob_rotate:
        return True

    fallback_only_keywords = ("button", "knob")
    for _, gt_steps in task_items:
        for gt_step in gt_steps:
            link_name = gt_step[1].split()[0]
            link_type = link_types.get(link_name, "")
            if not any(keyword in link_type for keyword in fallback_only_keywords):
                return True
    return False


def benchmark(args, manual_subdir, result_dict, view="front", img_size=1024):
    global total_planning, success_planning, total_execution, success_execution

    sample_dir = get_source_sample_dir(args, manual_subdir)
    sample_metadata = infer_sample_metadata(sample_dir)
    shape_id = sample_metadata["shape_id"]
    runtime_sample_dir = get_runtime_sample_dir(args, manual_subdir)
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
    manual_pngs_dir, _, manual_content = resolve_manual(
        runtime_sample_dir,
        manual_pdf_file,
        manual_source_dir=sample_dir,
    )
    page_vis_info_dict = None
    if not args.use_gt_alignment:
        manual_vis_info_path = os.path.join(manual_pngs_dir, f"manual_vis_info_{args.vlm_version}.json")
        legacy_manual_vis_info_path = os.path.join(manual_pngs_dir, "manual_vis_info.json")
        if os.path.exists(manual_vis_info_path) or os.path.exists(legacy_manual_vis_info_path):
            page_vis_info_dict = vis_analyze_page(manual_pngs_dir, vlm_version=args.vlm_version)

    env = None
    cam = None
    flog = open(args.log_path, "a", encoding="utf-8")
    if args.use_gt_alignment:
        pred_link_function_dict = load_gt_function_alignment(sample_dir)
        flog.write("Using ground-truth Track 3 function alignment\n")
    else:
        try:
            pred_link_function_dict = align_parts_nocad(
                sample_dir,
                runtime_sample_dir,
                sample_metadata,
                manual_pngs_dir,
                page_vis_info_dict,
                use_cache=not args.no_cache_alignment,
            )
        except RuntimeError:
            init_pose = load_init_pose(sample_dir, runtime_sample_dir)
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
                runtime_sample_dir,
                sample_metadata,
                manual_pngs_dir,
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

    init_pose = load_init_pose(sample_dir, runtime_sample_dir)
    link_types = None
    all_masks = None
    if args.execute:
        link_types = load_semantics(shape_id, args.data_dir)
    if args.execute and track3_execution_needs_masks(args, task_items, link_types):
        runtime_masks_path = os.path.join(runtime_sample_dir, "all_masks_track3.pkl")
        source_masks_path = os.path.join(sample_dir, "all_masks_track3.pkl")
        masks_path = runtime_masks_path if os.path.exists(runtime_masks_path) else source_masks_path
        if not os.path.exists(masks_path):
            tmp_env = env
            tmp_cam = cam
            created_tmp_env = False
            try:
                if tmp_env is None or tmp_cam is None:
                    tmp_env, tmp_cam = build_env_and_camera(
                        args,
                        shape_id,
                        init_pose,
                        args.obj_init_scale,
                        view,
                        img_size,
                        flog,
                    )
                    created_tmp_env = True
                masks_path = ensure_track3_masks(
                    sample_dir,
                    runtime_sample_dir,
                    sample_metadata,
                    tmp_env,
                    tmp_cam,
                )
            finally:
                if created_tmp_env:
                    cleanup_env_resources(env=tmp_env, cam=tmp_cam)
        with open(masks_path, "rb") as file:
            all_masks = pickle.load(file)
        if env is not None or cam is not None:
            cleanup_env_resources(env=env, cam=cam)
            env = None
            cam = None

    for task_idx, (task_name, gt_steps) in enumerate(task_items, start=1):
        processed_gt_steps = normalize_track3_gt_steps(gt_steps)
        if args.use_gt_plan:
            planned_steps = list(processed_gt_steps)
            flog.write("Using ground-truth plan for Track 3 planning\n")
        else:
            planned_steps = track3_plan_steps(
                sample_dir,
                runtime_sample_dir,
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
                runtime_sample_dir,
                shape_id,
                task_idx,
                gt_steps,
                exec_steps,
                pred_link_function_dict,
                init_pose,
                link_types,
                all_masks,
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
    cleanup_env_resources(env=env, cam=cam)
    return result_dict


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--manual_dir",
        default=os.path.join(ROOT_DIR, "data", "CheckManual_Data"),
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
    parser.add_argument("--use_gt_alignment", action="store_true", default=False)
    parser.add_argument("--use_gt_plan", action="store_true", default=False)
    parser.add_argument("--no_cache_alignment", action="store_true", default=False)
    parser.add_argument("--no_cache_plan", action="store_true", default=False)
    parser.add_argument("--save_vis", action="store_true", default=False)
    parser.add_argument("--button_camera_size", default=1024, type=int)
    parser.add_argument("--exec_camera_size", default=1536, type=int)
    parser.add_argument("--max_action_seconds", default=120.0, type=float)
    parser.add_argument("--try_physical_button_press", action="store_true", default=False)
    parser.add_argument("--try_physical_knob_rotate", action="store_true", default=False)
    parser.add_argument(
        "--no_sample_subprocess",
        action="store_true",
        default=False,
        help="Run samples in the current process instead of isolating Track 3 execution per sample.",
    )
    parser.add_argument("--_sample_worker", action="store_true", default=False, help=SUPPRESS)
    return parser.parse_args()


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)


def record_sample_error(args, result_dict, manual_subdir, error_message, details=None):
    sample_dir = os.path.join(args.manual_dir, manual_subdir)
    with open(args.log_path, "a", encoding="utf-8") as flog:
        flog.write(f"sample: {manual_subdir}\n")
        flog.write(f"Sample-level exception: {error_message}\n")
        if details:
            flog.write(details.rstrip() + "\n")
        flog.write("#########################################################\n")
        flog.write("----------------------------------------------------------------\n\n")

    result_dict[sample_dir] = {
        "total_tasks": [],
        "success_task_plan": [],
        "success_task_execution": [],
        "completion_rates": [],
        "pred_link_function_dict": None,
        "sample_error": error_message,
    }
    write_json(args.result_path, result_dict)


def run_sample_worker(args, manual_subdir):
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--manual_dir",
        args.manual_dir,
        "--data_dir",
        args.data_dir,
        "--out_dir",
        args.out_dir,
        "--log_path",
        args.log_path,
        "--result_path",
        args.result_path,
        "--sample",
        manual_subdir,
        "--max_samples",
        "1",
        "--_sample_worker",
        "--no_sample_subprocess",
    ]

    optional_string_args = [
        ("--global_tasks_path", args.global_tasks_path),
        ("--global_tasks_key", args.global_tasks_key),
        ("--task_name", args.task_name),
        ("--llm_version", args.llm_version),
        ("--vlm_version", args.vlm_version),
    ]
    for flag, value in optional_string_args:
        if value is not None:
            command.extend([flag, value])

    optional_value_args = [
        ("--max_tasks", args.max_tasks),
        ("--obj_init_scale", args.obj_init_scale),
        ("--button_camera_size", args.button_camera_size),
        ("--exec_camera_size", args.exec_camera_size),
        ("--max_action_seconds", args.max_action_seconds),
    ]
    for flag, value in optional_value_args:
        if value is not None:
            command.extend([flag, str(value)])

    flag_args = [
        ("--gui", args.gui),
        ("--execute", args.execute),
        ("--use_gt_alignment", args.use_gt_alignment),
        ("--use_gt_plan", args.use_gt_plan),
        ("--no_cache_alignment", args.no_cache_alignment),
        ("--no_cache_plan", args.no_cache_plan),
        ("--save_vis", args.save_vis),
        ("--try_physical_button_press", args.try_physical_button_press),
        ("--try_physical_knob_rotate", args.try_physical_knob_rotate),
    ]
    for flag, enabled in flag_args:
        if enabled:
            command.append(flag)

    return subprocess.run(command, capture_output=True, text=True, check=False)


def should_use_sample_subprocess(args):
    return args.execute and not args.no_sample_subprocess and not args._sample_worker


def tail_text(text, max_chars=4000):
    if not text:
        return ""
    return text[-max_chars:]


def summarize_subprocess_failure(result):
    details = []
    stdout_tail = tail_text(result.stdout)
    stderr_tail = tail_text(result.stderr)
    if stdout_tail:
        details.append("Subprocess stdout tail:\n" + stdout_tail)
    if stderr_tail:
        details.append("Subprocess stderr tail:\n" + stderr_tail)
    return "\n".join(details)


def summarize_rates_from_results(result_dict):
    planning_total = 0
    planning_success = 0
    execution_total = 0
    execution_success = 0

    for sample_result in result_dict.values():
        planning_total += len(sample_result.get("total_tasks", []))
        planning_success += len(sample_result.get("success_task_plan", []))
        completion_rates = sample_result.get("completion_rates", [])
        execution_total += len(completion_rates)
        execution_success += len(sample_result.get("success_task_execution", []))

    return planning_total, planning_success, execution_total, execution_success


if __name__ == "__main__":
    args = parse_args()
    args.out_dir = normalize_out_dir(args.out_dir)
    ensure_dir(args.out_dir)
    args.runtime_cache_dir = ensure_dir(os.path.join(args.out_dir, "runtime_cache"))
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
        if should_use_sample_subprocess(args):
            result = run_sample_worker(args, manual_subdir)
            if os.path.exists(args.result_path):
                result_dict = load_json(args.result_path)
            if result.returncode != 0:
                error_message = f"subprocess exited with return code {result.returncode}"
                if sample_dir not in result_dict:
                    record_sample_error(
                        args,
                        result_dict,
                        manual_subdir,
                        error_message,
                        summarize_subprocess_failure(result),
                    )
                else:
                    with open(args.log_path, "a", encoding="utf-8") as flog:
                        flog.write(f"sample: {manual_subdir}\n")
                        flog.write(f"Sample subprocess warning: {error_message}\n")
                        details = summarize_subprocess_failure(result)
                        if details:
                            flog.write(details.rstrip() + "\n")
                        flog.write("#########################################################\n")
                        flog.write("----------------------------------------------------------------\n\n")
            continue
        try:
            result_dict = benchmark(args, manual_subdir, result_dict)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            record_sample_error(
                args,
                result_dict,
                manual_subdir,
                f"{type(exc).__name__}: {exc}",
                traceback.format_exc(),
            )
            continue
        write_json(args.result_path, result_dict)

    if should_use_sample_subprocess(args):
        total_planning, success_planning, total_execution, success_execution = summarize_rates_from_results(result_dict)

    if total_planning > 0:
        print(f"Track 3 planning SR: {success_planning / total_planning:.4f}")
    if total_execution > 0:
        print(f"Track 3 execution SR: {success_execution / total_execution:.4f}")
