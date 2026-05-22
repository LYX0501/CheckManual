import os
import math
import re
import sys
import time
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
from utils import get_global_position_from_camera, save_h5, get_random_number, export_pts, render_pts_label_png
import cv2
import json
from argparse import ArgumentParser
from sapien.core import Pose
from env import Env, ContactError
from camera import Camera
from robots.panda_robot_gripper import Robot
from api_utils.gpt_api import *
from api_utils.ocr_api import *
from collections import Counter
import requests
import subprocess
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
import imageio
from scipy.spatial.transform import Rotation as R
from track1_ManualPlan import (
    get_camera_angles,
    extract_gt_link_function,
    resolve_manual,
    capture_link_photos,
    vis_analyze_page,
    align_parts,
    plan_steps,
)
from manualplan_support.manualplan_utils import (
    ROOT_DIR,
    ensure_dir,
    find_manual_pdf,
    find_part_state_file,
    get_object_urdf_path,
    get_robot_gripper_urdf_path,
    list_manual_subdirs,
    load_json,
    load_semantics,
    load_task_dict,
    normalize_out_dir,
    normalize_results_file,
)
import traceback

total_alignment, success_alignment = 0, 0
total_planning, success_planning = 0, 0
success_execution = 0


def require_open3d():
    try:
        import open3d as o3d  # type: ignore
    except ImportError:
        import open3d_compat as o3d  # type: ignore
    return o3d


def require_pybullet():
    try:
        import pybullet as p  # type: ignore
        import pybullet_data  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pybullet is required for this Track 2 action path. Install it in the "
            "checkmanual environment before running slider/door manipulation."
        ) from exc
    return p, pybullet_data

# result_dict = {}


def get_source_sample_dir(args, manual_subdir):
    return os.path.join(args.manual_dir, manual_subdir)


def get_runtime_sample_dir(args, manual_subdir):
    return ensure_dir(os.path.join(args.runtime_cache_dir, manual_subdir))


def sample_has_manual_cache(sample_dir):
    manual_pngs_dir = os.path.join(sample_dir, "manual_pngs")
    return os.path.exists(os.path.join(manual_pngs_dir, "manual_ocr_result.json")) or os.path.exists(
        os.path.join(manual_pngs_dir, "manual_text_fallback.txt")
    )


def sample_has_link_png_cache(sample_dir, link_count):
    link_pngs_dir = os.path.join(sample_dir, "link_pngs")
    if not os.path.isdir(link_pngs_dir):
        return False
    link_pngs = [file_name for file_name in os.listdir(link_pngs_dir) if file_name.endswith(".png")]
    return len(link_pngs) == link_count


def sample_has_alignment_cache(sample_dir, llm_version):
    link_pngs_dir = os.path.join(sample_dir, "link_pngs")
    candidates = [
        os.path.join(link_pngs_dir, f"pred_link_function_{llm_version}.json"),
        os.path.join(link_pngs_dir, "pred_link_function.json"),
    ]
    return any(os.path.exists(path) for path in candidates)


def sample_has_plan_cache(sample_dir, task_idx, llm_version):
    candidates = [
        os.path.join(sample_dir, f"task_{task_idx}_ManualPlan_{llm_version}_plans.json"),
        os.path.join(sample_dir, f"task_{task_idx}_plans.json"),
        os.path.join(sample_dir, "task_plans.json"),
    ]
    return any(os.path.exists(path) for path in candidates)


def rotation_matrix_to_quaternion(matrix):
    rotation = R.from_matrix(matrix)
    quaternion = rotation.as_quat()
    # 重新排列为 (w, x, y, z)，即实部在前
    quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
    
    return quaternion


def estimate_pose(args, shape_id, manual_subdir, view="front", img_size=1024):
    category = manual_subdir.replace(f"{shape_id}_", "").replace("_", "")
    source_sample_dir = get_source_sample_dir(args, manual_subdir)
    runtime_sample_dir = get_runtime_sample_dir(args, manual_subdir)
    pred_obj_pose_path = os.path.join(runtime_sample_dir, "pred_obj_pose.json")
    legacy_pred_obj_pose_path = os.path.join(source_sample_dir, "pred_obj_pose.json")
    if os.path.exists(pred_obj_pose_path) and not args.no_cache_pose:
        return load_json(pred_obj_pose_path)
    if os.path.exists(legacy_pred_obj_pose_path) and not args.no_cache_pose:
        return load_json(legacy_pred_obj_pose_path)

    flog = open(args.log_path, "a", encoding="utf-8")
    env = Env(flog=flog, show_gui=args.gui)

    object_urdf_fn = get_object_urdf_path(shape_id, args.data_dir)
    flog.write(f"object_urdf_fn: {object_urdf_fn}\n")
    object_material = env.get_material(4, 4, 0.01)

    flog.write("Object State: closed\n")
    env.load_object(object_urdf_fn, object_material, state="closed", rotation="front")

    phi, theta = get_camera_angles(view)
    cam = Camera(env, image_size=img_size, dist=5.0, phi=phi, theta=theta, fixed_position=True)
    cam_metadata = cam.get_metadata()

    env.step()
    env.render()
    rgb, depth = cam.get_observation()
    marked_rgb = (rgb * 255).astype(np.uint8)
    object_full_link_ids = [link.get_id() for link in env.object.get_links()]
    fallback_obj_mask = cam.get_movable_link_mask(object_full_link_ids)
    img_path = os.path.join(runtime_sample_dir, "rgb_track2.png")
    Image.fromarray(marked_rgb).save(img_path)

    command = [
        sys.executable,
        os.path.join(ROOT_DIR, "perception", "call_cv_server_crop.py"),
        img_path,
        category,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        flog.write("call_cv_server_crop.py failed; falling back to simulator mask\n")
        if result.stderr:
            flog.write(result.stderr + "\n")

    mask_path = os.path.join(runtime_sample_dir, "obj_mask_track2.npy")
    legacy_mask_path = os.path.join(source_sample_dir, "obj_mask_track2.npy")
    if not os.path.exists(mask_path) and os.path.exists(legacy_mask_path):
        mask_path = legacy_mask_path
    if os.path.exists(mask_path):
        obj_mask = np.load(mask_path)
    else:
        flog.write("Missing cropped object mask; using simulator-derived object mask\n")
        obj_mask = fallback_obj_mask
        np.save(os.path.join(runtime_sample_dir, "obj_mask_track2.npy"), obj_mask)

    data = {
        "cam_metadata": {
            "camera_matrix": cam_metadata["camera_matrix"].tolist(),
            "near": cam_metadata["near"],
            "far": cam_metadata["far"],
            "model_matrix": cam_metadata["model_matrix"].tolist(),
            "mat44": cam_metadata["mat44"].tolist(),
        },
        "rgbdmask": {
            "rgb": rgb.tolist(),
            "depth": depth.tolist(),
            "obj_mask": obj_mask.tolist(),
        },
        "path": os.path.join(args.data_dir, str(shape_id)),
    }

    pred_obj_pose = {"init_pos": [0, 0, 0], "init_rot": [1, 0, 0, 0]}
    for _ in range(5):
        try:
            response = requests.post(args.foundationpose_url, json=data, timeout=120)
            response.raise_for_status()
            foundationpose_pred = response.json()

            obj2world_urdf = cam_metadata["model_matrix"]
            obj2world_sim = np.array(foundationpose_pred["obj2world"])
            pred_world_urdf2world_sim = obj2world_sim @ np.linalg.inv(obj2world_urdf)
            pred_quaternion = rotation_matrix_to_quaternion(pred_world_urdf2world_sim[:3, :3])
            pred_position = list(pred_world_urdf2world_sim[:3, 3])
            pred_obj_pose = {"init_pos": pred_position, "init_rot": pred_quaternion}
            break
        except Exception:
            print("Request Error")

    with open(pred_obj_pose_path, "w", encoding="utf-8") as file:
        json.dump(pred_obj_pose, file, indent=4)

    flog.close()
    env.close()
    return pred_obj_pose

def push_button(flog, env, args, save_dir, shape_id, link_name, part_name, prev_joint_angles, view, img_size):
    success_flag = False
    
    link_nameid_dict = {name:id for name, id in zip(env.movable_link_names, env.movable_link_ids)}
    env.object.set_qpos(prev_joint_angles)
    
    phi, theta = get_camera_angles(view)
    camera_size = max(img_size, 2048)
    cam = Camera(env, dist=5, image_size=camera_size, phi=phi, theta=theta)
    
    robot_urdf_fn = args.robot_urdf_path
    robot_material = env.get_material(4, 4, 0.01)
    robot_init_scale = 1
    robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=True, scale=robot_init_scale)
    
    env.step()
    env.render()
    rgb, depth = cam.get_observation()

    target_link_id = link_nameid_dict[link_name]
    
    print(f"Part Name: {part_name}")
    env.set_target_object_part_actor_id(target_link_id)
    
    # 此处暂时用GT Mask来打通流程
    link_mask = cam.get_movable_link_mask([target_link_id])
    
    xs, ys = np.where(link_mask>0)
    if len(xs) == 0:
        env.scene.remove_articulation(robot.robot)
        return prev_joint_angles, success_flag
    x_center = int(np.mean(xs))
    y_center = int(np.mean(ys))
    x, y = x_center, y_center

    # get pixel 3D pulling direction (cam/world)
    gt_nor = cam.get_normal_map()
    direction_cam = gt_nor[x, y, :3]
    direction_cam /= np.linalg.norm(direction_cam)
    action_direction_cam = -direction_cam
    action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ action_direction_cam

    cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
    cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
    position_cam = cam_XYZA[x, y, :3]
    position_cam_xyz1 = np.ones((4), dtype=np.float32)
    position_cam_xyz1[:3] = position_cam
    position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
    position_world = position_world_xyz1[:3]

    # compute final pose
    up = np.array(action_direction_world, dtype=np.float32)  # up = action_direction_world
    camera_frame = cam.get_metadata()["mat44"][:3, :3]
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
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)  # get unit vector
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward)
    
    rotmat = np.eye(4).astype(np.float32)
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up

    start_rotmat = np.array(rotmat, dtype=np.float32)
    start_rotmat[:3, 3] = position_world - action_direction_world * 0.5
    start_pose = Pose().from_transformation_matrix(start_rotmat)
    robot.robot.set_root_pose(start_pose)

    robot.close_gripper()
    env.render()
    
    rgb, depth = cam.get_observation()
    marked_rgb = (rgb * 255).astype(np.uint8)
    
    print('Move. Now wait.')

    init_target_part_qpos = env.get_target_part_qpos()

    cv2.imwrite(os.path.join(save_dir, "start_pose.png"), marked_rgb)

    try:
        apprchimgs = []
        button_attempts = [
            (0.08, 2200, 400),
            (0.05, 700, 250),
        ]
        abs_motion = 0.0
        tot_motion = 0.0
        for attempt_idx, (final_dist, move_steps, settle_steps) in enumerate(button_attempts, start=1):
            final_rotmat = np.array(rotmat, dtype=np.float32)
            final_rotmat[:3, 3] = position_world - action_direction_world * final_dist
            if attempt_idx == 1:
                apprchimgs = robot.move_to_target_pose(
                    final_rotmat,
                    move_steps,
                    cam=cam,
                    vis_gif=True,
                    vis_gif_interval=100,
                    visu=False,
                )
            else:
                robot.move_to_target_pose(final_rotmat, move_steps, cam=cam, vis_gif=False, visu=False)

            print("Moved to target pose")
            robot.close_gripper()
            robot.wait_n_steps(settle_steps)

            final_target_part_qpos = env.get_target_part_qpos()
            abs_thres = 0.01
            rel_thres = 0.5
            abs_motion = abs(final_target_part_qpos - init_target_part_qpos)
            j = env.target_object_part_joint_id
            tot_motion = env.joint_angles_upper[j] - env.joint_angles_lower[j] + 1e-8
            success_flag = (abs_motion > abs_thres) or (abs_motion / tot_motion > rel_thres)
            flog.write(
                f"{shape_id}_{link_name}_{part_name} attempt_{attempt_idx} "
                f"abs_motion: {abs_motion} ; tot_motion: {tot_motion} ; success: {success_flag}\n"
            )
            if success_flag:
                break

        env.render()
        rgb_final_pose, _ = cam.get_observation()
        rgb_final_pose = (rgb_final_pose * 255).astype(np.uint8)
        rgb_final_pose = cv2.circle(rgb_final_pose, (y, x), radius=4, color=(0, 0, 255), thickness=5)

        Image.fromarray(rgb_final_pose).save(os.path.join(save_dir, "target_pose.png"))
        imageio.mimsave(os.path.join(save_dir, "apprchimgs.gif"), apprchimgs)
    except Exception as exc:
        flog.write(f"push_button exception: {exc}\n")
        flog.write(traceback.format_exc() + "\n")
    
    updated_joint_angles = env.get_object_qpos()
    print("after updated_joint_angles")
    
    env.scene.remove_articulation(robot.robot)
    # flog.close()
    # env.close()
    
    return updated_joint_angles, success_flag


def rotate_knob(flog, env, args, save_dir, shape_id, link_name, part_name, rotation_degrees, prev_joint_angles, view, img_size):
    success_flag = False
    
    link_nameid_dict = {name:id for name, id in zip(env.movable_link_names, env.movable_link_ids)}
    env.object.set_qpos(prev_joint_angles)
    
    phi, theta = get_camera_angles(view)
    # cam = Camera(env, dist=10, image_size=img_size, phi=phi, theta=theta)
    cam = Camera(env, dist=5, image_size=4096, phi=phi, theta=theta)
    
    # print("cam", dir(cam.camera))
    robot_urdf_fn = args.robot_urdf_path
    robot_material = env.get_material(4, 4, 0.01)
    robot_init_scale = 1.0
    robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=True, scale=robot_init_scale)

    env.step()
    env.render()
    rgb, depth = cam.get_observation()
    
    target_link_id = link_nameid_dict[link_name]
    
    print(f"Part Name: {part_name}")
    env.set_target_object_part_actor_id(target_link_id)
    
    link_mask = cam.get_movable_link_mask([target_link_id])
    xs, ys = np.where(link_mask>0)
    if len(xs) == 0:
        print("len(xs) == 0")
        env.scene.remove_articulation(robot.robot)
        return prev_joint_angles, success_flag
    
    idx = np.random.randint(len(xs)) # Randomly sample a pixel to interact
    x, y = xs[idx], ys[idx]
    
    # Calculate the geometric center of the pixels where link_mask > 0
    x_center = int(np.mean(xs))
    y_center = int(np.mean(ys))
    x, y = x_center, y_center

    # get pulling direction (cam/world)
    print("Calculate action direction")
    gt_nor = cam.get_normal_map()
    direction_cam = gt_nor[x, y, :3]
    direction_cam /= np.linalg.norm(direction_cam)
    action_direction_cam = -direction_cam
    action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ action_direction_cam
    

    print("Calculate manip position")
    cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
    cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
    position_cam = cam_XYZA[x, y, :3]
    position_cam_xyz1 = np.ones((4), dtype=np.float32)
    position_cam_xyz1[:3] = position_cam
    position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
    position_world = position_world_xyz1[:3]

    # compute final pose
    print("Calculate rotmat")
    up = np.array(action_direction_world, dtype=np.float32)  # up = action_direction_world
    forward = np.random.randn(3).astype(np.float32)
    forward /= np.linalg.norm(forward)
    while abs(up @ forward) > 0.99:
        forward = np.random.randn(3).astype(np.float32)
        forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)  # get unit vector
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward)
    
    rotmat = np.eye(4).astype(np.float32)
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up

    start_rotmat = np.array(rotmat, dtype=np.float32)
    start_rotmat[:3, 3] = position_world - action_direction_world * 0.5
    start_pose = Pose().from_transformation_matrix(start_rotmat)
    print("Set start gripper pose")
    robot.robot.set_root_pose(start_pose)

    final_dist = 0
    final_rotmat = np.array(rotmat, dtype=np.float32)
    final_rotmat[:3, 3] = position_world - action_direction_world * final_dist # Minus half-length of gripper

    robot.open_gripper()
    robot.wait_n_steps(800)
    env.step()
    env.render()
    
    rgb, depth = cam.get_observation()
    marked_rgb = (rgb * 255).astype(np.uint8)

   
    cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
    o3d = require_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cam_XYZA_pts)
    valid_indices = np.where(depth < 1)
    pcd.colors = o3d.utility.Vector3dVector(rgb[valid_indices])
    o3d.io.write_point_cloud("point_cloud.ply", pcd)
    

    
    cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1]) 
            
    init_target_part_qpos = env.get_target_part_qpos()

    # save_dir = os.path.join(out_dir, f"{shape_id}_{link_name}")
    # os.mkdir(save_dir)

    marked_rgb = cv2.circle(marked_rgb, (y, x), radius=4, color=(0, 0, 255), thickness=5)
    cv2.imwrite(os.path.join(save_dir, "start_pose.png"), marked_rgb)

    print('Move to target pose.')
    try:
        apprchimgs = robot.move_to_target_pose(final_rotmat, 3000, cam=cam, vis_gif=True, vis_gif_interval=100, visu=False)
    except Exception as e:
        print(f"Error occurred at line {traceback.extract_tb(e.__traceback__)[-1].lineno}: {e}")
        env.scene.remove_articulation(robot.robot)
        return prev_joint_angles, success_flag
    
    robot.wait_n_steps(2000)
    
    print("Close gripper")
    robot.close_gripper()
    robot.wait_n_steps(800)
    
    now_qpos = robot.robot.get_qpos().tolist()
    finger1_qpos = now_qpos[-1]
    finger2_qpos = now_qpos[-2]
    if finger1_qpos + finger2_qpos > 0.01:
        success_grasp = True
    else:
        print("finger1_qpos + finger2_qpos", finger1_qpos + finger2_qpos)
        success_grasp = False
    
    
    env.render()
    rgb_final_pose, _ = cam.get_observation()
    rgb_final_pose = (rgb_final_pose * 255).astype(np.uint8)
    rgb_final_pose = cv2.circle(rgb_final_pose, (y, x), radius=4, color=(0, 0, 255), thickness=5)
    
    Image.fromarray(rgb_final_pose).save(os.path.join(save_dir, "target_pose.png"))
    imageio.mimsave(os.path.join(save_dir, "apprchimgs.gif"), apprchimgs)

    final_target_part_qpos = env.get_target_part_qpos()
    # TASK SUCCESS GOAL
    abs_motion = abs(final_target_part_qpos - init_target_part_qpos)
    abs_degrees = round(math.degrees(abs_motion), 2)
    j = env.target_object_part_joint_id
    tot_motion = env.joint_angles_upper[j] - env.joint_angles_lower[j] + 1e-8
    
    flog.write(f"{shape_id}_{link_name}_{part_name} Anygrasp Success: {success_grasp} ; before rotation abs_motion: {abs_motion} ({abs_degrees}°) \n")

    print("Start rotation")

    theta = 2 * np.pi * (rotation_degrees/360)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # 构建围绕 up 轴的旋转矩阵
    R = np.array([
        [cos_theta + up[0]**2 * (1 - cos_theta), up[0]*up[1]*(1 - cos_theta) - up[2]*sin_theta, up[0]*up[2]*(1 - cos_theta) + up[1]*sin_theta],
        [up[1]*up[0]*(1 - cos_theta) + up[2]*sin_theta, cos_theta + up[1]**2 * (1 - cos_theta), up[1]*up[2]*(1 - cos_theta) - up[0]*sin_theta],
        [up[2]*up[0]*(1 - cos_theta) - up[1]*sin_theta, up[2]*up[1]*(1 - cos_theta) + up[0]*sin_theta, cos_theta + up[2]**2 * (1 - cos_theta)]
    ], dtype=np.float32)

    # 将新旋转矩阵应用于现有的旋转矩阵
    rotmat[:3, :3] = R @ rotmat[:3, :3]

    # Move the robot to the new rotated target pose
    try:
        apprchimgs_rotated = robot.move_to_target_pose(rotmat, 3000, cam=cam, vis_gif=True, vis_gif_interval=100, visu=False)
    except Exception as e:
        print(f"Error occurred at line {traceback.extract_tb(e.__traceback__)[-1].lineno}: {e}")
        # env.close()
        # flog.close()
        updated_joint_angles = env.get_object_qpos()
        return updated_joint_angles, success_flag
    
    # Wait for the robot to finish moving
    robot.wait_n_steps(2000)

    # Capture the final pose after rotation
    env.render()
    rgb_final_pose_rotated, _ = cam.get_observation()
    rgb_final_pose_rotated = (rgb_final_pose_rotated * 255).astype(np.uint8)
    rgb_final_pose_rotated = cv2.circle(rgb_final_pose_rotated, (y, x), radius=4, color=(0, 0, 255), thickness=5)

    # Save the final pose image after rotation
    Image.fromarray(rgb_final_pose_rotated).save(os.path.join(save_dir, "target_pose_rotated.png"))

    # # Save the approach images as a GIF
    imageio.mimsave(os.path.join(save_dir, "apprchimgs_rotated.gif"), apprchimgs_rotated)

    print("Complete manipulation")
    
    final_target_part_qpos = env.get_target_part_qpos()
    # TASK SUCCESS GOAL
    abs_motion = abs(final_target_part_qpos - init_target_part_qpos)
    abs_degrees = round(math.degrees(abs_motion), 2)
    j = env.target_object_part_joint_id
    tot_motion = env.joint_angles_upper[j] - env.joint_angles_lower[j] + 1e-8

    flog.write(f"{shape_id}_{link_name}_{part_name} after rotation abs_motion: {abs_motion} ({abs_degrees}°) \n\n")
            
    
    if abs_degrees > rotation_degrees - 30 and abs_degrees < rotation_degrees + 30:
        success_flag = True     
    
    updated_joint_angles = env.get_object_qpos()
    env.scene.remove_articulation(robot.robot)
    
    return updated_joint_angles, success_flag
    
    
def send_ply_file(file_path, server_url, lims):
    with open(file_path, 'rb') as file:
        filename = os.path.basename(file_path)
        files = {'file': (filename, file, 'application/octet-stream')}
        data = {'lims': json.dumps(lims), 'scale': 1}
        
        print("Transfer ply file")
        response = requests.post(server_url, files=files, data=data, timeout=120)
    return response

def rotate_around_axis(target_gg_rotmat, joint_world_position, joint_axis_world, angle_degrees):
    # 将角度转换为弧度
    angle_rad = np.deg2rad(angle_degrees)
    
    # 创建旋转矩阵（使用Rodrigues公式或scipy的Rotation）
    rotation = R.from_rotvec(angle_rad * joint_axis_world)
    rotation_matrix = rotation.as_matrix()
    
    # 提取原始位置和旋转
    original_position = target_gg_rotmat[:3, 3]
    original_rotation = target_gg_rotmat[:3, :3]
    
    # 计算相对于旋转轴的位置
    relative_position = original_position - joint_world_position
    
    # 旋转相对位置
    rotated_relative_position = np.dot(rotation_matrix, relative_position)
    
    # 计算新的绝对位置
    new_position = rotated_relative_position + joint_world_position
    
    # 计算新的旋转矩阵
    new_rotation = np.dot(rotation_matrix, original_rotation)
    
    # 构建新的4x4变换矩阵
    new_transform = np.eye(4)
    new_transform[:3, :3] = new_rotation
    new_transform[:3, 3] = new_position
    
    return new_transform
    
def manip_slider_or_door(flog, env, args, save_dir, manual_subdir, shape_id, link_name, part_name, prev_joint_angles, joint2world_dict, view, img_size, use_cached_grasp=False):
    success_flag = False
    o3d = require_open3d()
    p, pybullet_data = require_pybullet()

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    try:
        object_urdf_fn = get_object_urdf_path(shape_id, args.data_dir)
        obj_urdf = p.loadURDF(object_urdf_fn)
    except:
        return prev_joint_angles, success_flag
    
    numJoints = p.getNumJoints(obj_urdf)
    linkname_jointidx_dict = {}
    for joint_idx in range(numJoints):
        joint_info = p.getJointInfo(obj_urdf, joint_idx)
        joint_child_link = joint_info[12].decode('utf-8')
        linkname_jointidx_dict[joint_child_link] = joint_idx
    
    current_joint_idx = linkname_jointidx_dict[link_name]
    initial_link_state = p.getLinkState(obj_urdf, current_joint_idx)

    # 获取关节轴
    joint_world_position = initial_link_state[0]
    joint_world_orientation = initial_link_state[1]
    joint_info = p.getJointInfo(obj_urdf, current_joint_idx)
    joint_axis_local = joint_info[13]  # 关节轴在局部坐标系下
    world_rotation_matrix = p.getMatrixFromQuaternion(joint_world_orientation)
    joint_axis_world = np.dot(np.array(world_rotation_matrix).reshape(3, 3), joint_axis_local)
    joint_axis_world = joint_axis_world / np.linalg.norm(joint_axis_world)
    
    
    
    # flog = open('benchmark_task2_log.txt', 'a')
    # env = Env(flog=flog, show_gui=(not args.no_gui)) 

    # # load object
    # object_urdf_fn = '../data/sapien_dataset/%s/mobility.urdf' % shape_id
    # flog.write('object_urdf_fn: %s\n' % object_urdf_fn)
    # object_material = env.get_material(4, 4, 0.01)

    # state = "closed"
    # flog.write('Object State: %s\n' % state)
    # link_dict = load_semantics(shape_id)
    # _ = env.load_object(object_urdf_fn, object_material, state=state, rotation="left")
    link_nameid_dict = {name:id for name, id in zip(env.movable_link_names, env.movable_link_ids)}
    env.object.set_qpos(prev_joint_angles)

    phi, theta = get_camera_angles(view)
    cam = Camera(env, image_size=img_size, dist=5.0, phi=phi, theta=theta, fixed_position=True)

    for _ in range(1000):
        env.step()
        env.render()

    # target_link_joint = joint2world_dict[link_name]
    target_link_id = link_nameid_dict[link_name]
    env.set_target_object_part_actor_id(target_link_id) 

    runtime_sample_dir = get_runtime_sample_dir(args, manual_subdir)
    cache_grasp_dir_path = os.path.join(runtime_sample_dir, "cache_grasp_poses")
    ensure_dir(cache_grasp_dir_path)

    cache_grasp_path = os.path.join(cache_grasp_dir_path, f"{shape_id}_{link_name}_{part_name}_anygrasp.json")
    source_sample_dir = get_source_sample_dir(args, manual_subdir)
    source_cache_grasp_path = os.path.join(
        source_sample_dir,
        "cache_grasp_poses",
        f"{shape_id}_{link_name}_{part_name}_anygrasp.json",
    )
    if use_cached_grasp and os.path.exists(cache_grasp_path):
        with open(cache_grasp_path, "r", encoding="utf-8") as file:
            grasp_data = json.load(file)
    elif use_cached_grasp and os.path.exists(source_cache_grasp_path):
        with open(source_cache_grasp_path, "r", encoding="utf-8") as file:
            grasp_data = json.load(file)
    else:
        views = ["right", "left", "front"]
        combined_pcd = o3d.geometry.PointCloud()  # 创建一个空的点云对象用于合并

        for view in views:
            phi, theta = get_camera_angles(view)
            # cam = Camera(env, dist=10, image_size=img_size, phi=phi, theta=theta)
            cam = Camera(env, dist=5, image_size=4096, phi=phi, theta=theta)
            
            env.step()
            env.render()

            rgb, depth = cam.get_observation()
            _, _, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
            
            # 创建点云并设置颜色
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cam_XYZA_pts)
            pcd.colors = o3d.utility.Vector3dVector(rgb[np.where(depth < 1)])

            # 将点云从相机坐标系转换到世界坐标系
            pcd.transform(cam.mat44)  # 使用 cam.mat44 进行变换

            # 合并点云
            combined_pcd += pcd
        
        # Transfer arget link bbox to lims
        target_link_mask = cam.get_movable_link_mask([target_link_id])
        handle_mask = cam.get_handle_mask()
        target_link_mask = handle_mask * target_link_mask if len(np.unique(handle_mask)) > 1 else target_link_mask
        if len(np.unique(target_link_mask)) > 1:
            depth[np.where(target_link_mask < 1)] = 1 
        _, _, cam_XYZA_pts = cam.compute_camera_XYZA(depth)

        min_x, min_y, min_z = np.min(cam_XYZA_pts[:, :3], axis=0)
        max_x, max_y, max_z = np.max(cam_XYZA_pts[:, :3], axis=0)

        bbox_points = np.array([
            [min_x, min_y, min_z],
            [max_x, max_y, max_z]
        ])
        homogeneous_bbox_points = np.hstack((bbox_points, np.ones((bbox_points.shape[0], 1))))
        bbox_points = np.dot(cam.mat44, homogeneous_bbox_points.T).T[:, :3]
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=np.float32)
        bbox_points_rotated = np.dot(bbox_points, rotation_matrix)

        theta = np.radians(-90)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ], dtype=np.float32)
        bbox_points = np.dot(rotation_matrix, bbox_points_rotated.T).T
        lims = list(bbox_points[0, :]) + list(bbox_points[1, :])

        voxel_size = 0.005
        grasp_data = {'gg_score': [], 'gg_pos': [], 'gg_rot': []}
        
        for _ in range (10):
            try:
                downsampled_pcd = combined_pcd.voxel_down_sample(voxel_size)
                print("Point Cloud Size: ", len(downsampled_pcd.points))
                save_ply_path = os.path.join(save_dir, f"{shape_id}_{link_name}_{part_name}_pcd.ply")
                o3d.io.write_point_cloud(save_ply_path, downsampled_pcd)

                # Send pcd to grasp server
                if not args.grasp_server_url:
                    raise RuntimeError(
                        "cache_grasp_poses is missing and --grasp_server_url was not provided."
                    )
                response = send_ply_file(save_ply_path, args.grasp_server_url, lims)
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")
                if response.text == "Empty grasp poses":
                    flog.write("Empty grasp poses\n")
                    return prev_joint_angles, success_flag
                else:
                    grasp_data = response.json()
                break
            except:
                voxel_size += 0.001

        
        with open(cache_grasp_path, 'w') as response_file:
            json.dump(grasp_data, response_file, indent=4)

    phi, theta = get_camera_angles("front")
    # cam = Camera(env, dist=10, image_size=img_size, phi=phi, theta=theta)
    cam = Camera(env, dist=5, image_size=4096, phi=phi, theta=theta)
    
    env.step()
    env.render()
    
    gg_score = grasp_data['gg_score']
    gg_pos = [np.array(item) for item in grasp_data['gg_pos']]
    gg_rot = [np.array(item) for item in grasp_data['gg_rot']]

    success_grasp = False
    robot_existence = False
    gg_len = len(gg_score)
    for gg_idx in range(gg_len):
        gg_idx_ = gg_idx + 1
        # flog = open('robot_log.txt', 'a')     
        flog.write(f"Try {gg_idx_}/{gg_len} gg\n")       
        

        robot_urdf_fn = args.robot_urdf_path
        robot_material = env.get_material(4, 4, 0.01)
        robot_init_scale = 1.0
        robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=True, scale=robot_init_scale)
        robot_existence = True
        
        # score = gg_score[gg_idx]
        # if score < 0.05: 
        #     raise ValueError("Low grasp score ", score)
        
        gg_rotmat = np.eye(4).astype(np.float32)
        gg_rotmat[:3, :3] = gg_rot[gg_idx]
        gg_rotmat[:3, 3] = gg_pos[gg_idx]
    
        # 定义绕 Y 轴的旋转矩阵
        theta_y = np.radians(90)
        rotation_matrix_y = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y), 0],
            [0, 1, 0, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y), 0],
            [0, 0, 0, 1]
        ])

        # 定义绕 X 轴的旋转矩阵
        theta_x = np.radians(90)
        rotation_matrix_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x), 0],
            [0, np.sin(theta_x), np.cos(theta_x), 0],
            [0, 0, 0, 1]
        ])
                    
        gg_rotmat = np.dot(rotation_matrix_y, gg_rotmat)
        target_gg_rotmat = np.dot(rotation_matrix_x, gg_rotmat)

        anygrasp_to_franka_rot = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
        target_gg_rotmat[:3, :3] = target_gg_rotmat[:3, :3] @ anygrasp_to_franka_rot
        offset_matrix = np.eye(4).astype(np.float32)
        offset_matrix[0, 3] = 0.1
        target_gg_rotmat = np.dot(offset_matrix, target_gg_rotmat)
        print("target_gg_rotmat", target_gg_rotmat)

        pre_matrix = np.eye(4).astype(np.float32)
        pre_matrix[0, 3] = -0.2
        pre_gg_rotmat = np.dot(pre_matrix, target_gg_rotmat)

        pre_gg_pose = Pose().from_transformation_matrix(pre_gg_rotmat)
        robot.robot.set_root_pose(pre_gg_pose)

        print("Open gripper")
        robot.open_gripper()
        robot.wait_n_steps(500)
        
    
        print("Move to target pose")
        apprchimgs = robot.move_to_target_pose(target_gg_rotmat, 1000, cam=cam, vis_gif=True, vis_gif_interval=100, visu=False)
        robot.wait_n_steps(500)
        
        print("Close gripper")
        robot.close_gripper()
        robot.wait_n_steps(500)
        
        now_qpos = robot.robot.get_qpos().tolist()
        finger1_qpos = now_qpos[-1]
        finger2_qpos = now_qpos[-2]
        finger_distance = finger1_qpos + finger2_qpos
        flog.write(f"finger1_qpos + finger2_qpos: {finger_distance}")
        if finger_distance > 0.015:
            success_grasp = True
            flog.write("Successful Grasp\n")
        else:
            flog.write("Fail to Grasp\n")
        

        env.step()
        env.render()
        rgb_gg_pose, _ = cam.get_observation()
        rgb_gg_pose = (rgb_gg_pose * 255).astype(np.uint8)
        Image.fromarray(rgb_gg_pose).save(os.path.join(save_dir, f"gg_pose_{gg_idx}.png"))
        imageio.mimsave(os.path.join(save_dir, f"apprchimgs_{gg_idx}.gif"), apprchimgs)
        
        if success_grasp:
            break
        else:
            env.scene.remove_articulation(robot.robot)
            robot_existence = False

    if success_grasp:
        if "door" in part_name:
            init_target_part_qpos = env.get_target_part_qpos()
            
            # pullup_matrix = np.eye(4).astype(np.float32)
            # pullup_matrix[2, 3] = 0.08
            # pullup_rotmat = np.dot(pullup_matrix, target_gg_rotmat)
            target_degrees = 60
            opendoor_rotmat = rotate_around_axis(target_gg_rotmat, joint_world_position, joint_axis_world, target_degrees)

            print("Open")
            pullupimgs = robot.move_to_target_pose(opendoor_rotmat, 4000, cam=cam, vis_gif=True, vis_gif_interval=100, visu=False)
            robot.wait_n_steps(1500)

            env.step()
            env.render()
            rgb_pullup, _ = cam.get_observation()
            rgb_pullup = (rgb_pullup * 255).astype(np.uint8)
            Image.fromarray(rgb_pullup).save(os.path.join(save_dir, f"pullup_{gg_idx}.png"))
            imageio.mimsave(os.path.join(save_dir, f"pullupimgs_{gg_idx}.gif"), pullupimgs)

            final_target_part_qpos = env.get_target_part_qpos()
            abs_motion = abs(final_target_part_qpos - init_target_part_qpos)
            abs_degrees = round(math.degrees(abs_motion), 2)
            print(f"Open the door abs_degrees: {abs_degrees}")
            flog.write(f"{shape_id}_{link_name}_{part_name} Open the door abs_degrees: {abs_degrees} \n")
            
            if abs_degrees > 30:
                success_flag = True
                print("Successfully open the door!")
                flog.write("Successfully open the door\n")        
            
        else:
            init_target_part_qpos = env.get_target_part_qpos()
            
            # pullup_matrix = np.eye(4).astype(np.float32)
            # pullup_matrix[2, 3] = 0.08
            # pullup_rotmat = np.dot(pullup_matrix, target_gg_rotmat)

            pullup_rotmat = target_gg_rotmat.copy()
            pullup_rotmat[:3, 3] = target_gg_rotmat[:3, 3] + joint_axis_world

            print("Pull up")
            pullupimgs = robot.move_to_target_pose(pullup_rotmat, 1000, cam=cam, vis_gif=True, vis_gif_interval=100, visu=False)
            robot.wait_n_steps(500)

            env.step()
            env.render()
            rgb_pullup, _ = cam.get_observation()
            rgb_pullup = (rgb_pullup * 255).astype(np.uint8)
            Image.fromarray(rgb_pullup).save(os.path.join(save_dir, f"pullup_{gg_idx}.png"))
            imageio.mimsave(os.path.join(save_dir, f"pullupimgs_{gg_idx}.gif"), pullupimgs)

            final_target_part_qpos = env.get_target_part_qpos()
            abs_motion = abs(final_target_part_qpos - init_target_part_qpos)
            print(f"Pull up abs_motion: {abs_motion}")
            flog.write(f"{shape_id}_{link_name}_{part_name} Pull up abs_motion: {abs_motion} \n")
            
            if abs_motion > 0.01:
                success_flag = True
                print("Successfully pull up!")
                flog.write("Successfully pull up!\n")

    print(4)
    flog.write("\n")
    updated_joint_angles = env.get_object_qpos()

    if robot_existence == True:
        env.scene.remove_articulation(robot.robot)
    
    return updated_joint_angles, success_flag
    
    
def benchmark(args, manual_subdir, result_dict, pred_obj_pose, obj_init_scale=0.4, view="front", img_size=1024):
    global total_alignment, success_alignment, total_planning, success_planning, success_execution

    shape_id = manual_subdir.split("_")[0]
    source_sample_dir = get_source_sample_dir(args, manual_subdir)
    runtime_sample_dir = get_runtime_sample_dir(args, manual_subdir)
    result_dict[source_sample_dir] = {
        "total_tasks": [],
        "success_task_plan": [],
        "success_task_execution": [],
        "completion_rates": [],
        "pred_link_function_dict": None,
    }

    print(f"Evaluate {manual_subdir}")

    manual_pdf_file = find_manual_pdf(source_sample_dir)
    task_dict, task_json_path = load_task_dict(
        source_sample_dir,
        global_tasks_path=args.global_tasks_path,
        global_key=args.global_tasks_key,
    )
    part_state_json_file = find_part_state_file(source_sample_dir)
    part_state_dict = load_json(os.path.join(source_sample_dir, part_state_json_file))
    gt_link_function_dict = extract_gt_link_function(part_state_dict)

    flog = open(args.log_path, "a", encoding="utf-8")
    env = Env(flog=flog, show_gui=args.gui)
    try:
        object_urdf_fn = get_object_urdf_path(shape_id, args.data_dir)
        flog.write(f"object_urdf_fn: {object_urdf_fn}\n")
        flog.write(f"task_json: {task_json_path}\n")
        object_material = env.get_material(4, 4, 0.01)

        flog.write("Object State: closed\n")
        link_dict = load_semantics(shape_id, args.data_dir)

        init_joint_angles = env.load_object(
            object_urdf_fn,
            object_material,
            state="closed",
            init_scale=obj_init_scale,
            init_pos=pred_obj_pose["init_pos"],
            init_rot=pred_obj_pose["init_rot"],
        )
        link_nameid_dict = {
            name: link_id
            for name, link_id in zip(env.movable_link_names, env.movable_link_ids)
            if name in gt_link_function_dict
        }

        phi, theta = get_camera_angles(view)
        cam = Camera(env, image_size=img_size, dist=5.0, phi=phi, theta=theta, fixed_position=True)
        joint2world_dict = {}

        manual_cache_dir = source_sample_dir if sample_has_manual_cache(source_sample_dir) else runtime_sample_dir
        link_cache_dir = (
            source_sample_dir
            if sample_has_link_png_cache(source_sample_dir, len(link_nameid_dict))
            else runtime_sample_dir
        )
        alignment_cache_dir = (
            source_sample_dir
            if sample_has_alignment_cache(source_sample_dir, args.llm_version)
            else runtime_sample_dir
        )

        manual_pngs_dir, _, manual_content = resolve_manual(
            manual_cache_dir,
            manual_pdf_file,
            manual_source_dir=source_sample_dir,
        )
        link_pngs_dir = capture_link_photos(env, link_cache_dir, link_nameid_dict, img_size)
        page_vis_info_dict = vis_analyze_page(manual_pngs_dir, vlm_version=args.vlm_version)
        pred_link_function_dict = align_parts(
            alignment_cache_dir,
            link_pngs_dir,
            page_vis_info_dict,
            llm_version=args.llm_version,
            vlm_version=args.vlm_version,
        )
        if args.use_gt_alignment:
            flog.write("Using ground-truth alignment for smoke test\n")
            pred_link_function_dict = gt_link_function_dict
        result_dict[source_sample_dir]["pred_link_function_dict"] = pred_link_function_dict

        total_alignment += 1
        if gt_link_function_dict == pred_link_function_dict:
            success_alignment += 1
            flog.write(
                f"{shape_id} Correct Alignment. Success Alignment Rate: "
                f"{success_alignment / total_alignment}\n"
            )
        else:
            flog.write(
                f"{shape_id} Incorrect Alignment. Success Alignment Rate: "
                f"{success_alignment / total_alignment}\n"
            )
            flog.write("Ground Truth Link Function Dict:\n")
            flog.write(f"{json.dumps(gt_link_function_dict, indent=4)}\n")
            flog.write("Predicted Link Function Dict:\n")
            flog.write(f"{json.dumps(pred_link_function_dict, indent=4)}\n")

        asset_save_dir = ensure_dir(os.path.join(args.out_dir, "track2", manual_subdir))
        task_items = list(task_dict.items())
        if args.max_tasks is not None:
            task_items = task_items[: args.max_tasks]

        for task_idx, (task_name, gt_steps_raw) in enumerate(task_items, start=1):
            task_save_dir = ensure_dir(os.path.join(asset_save_dir, f"task_{task_idx}"))
            gt_steps = [item[1].split()[0] + ": " + item[2] for item in gt_steps_raw]

            planned_steps = plan_steps(
                source_sample_dir if sample_has_plan_cache(source_sample_dir, task_idx, args.llm_version) else runtime_sample_dir,
                task_idx,
                manual_content,
                task_name,
                pred_link_function_dict,
                llm_version=args.llm_version,
                use_cache_plan=not args.no_cache_plan,
            )
            if args.use_gt_plan:
                flog.write("Using ground-truth plan for smoke test\n")
                planned_steps = list(gt_steps)
            total_planning += 1
            result_dict[source_sample_dir]["total_tasks"].append(task_name)
            flog.write(f"Task Name: {task_name}\n")

            if gt_steps == planned_steps:
                success_planning += 1
                result_dict[source_sample_dir]["success_task_plan"].append(task_name)
                flog.write(
                    f"{shape_id} Correct Planning. Success Planning Rate: "
                    f"{success_planning / total_planning}\n"
                )

            joint_angles = init_joint_angles
            success_flag = False
            step_idx = 0
            success_step_count = 0
            for action_step in planned_steps:
                if step_idx > len(gt_steps) - 1:
                    break
                gt_step = gt_steps[step_idx]
                if action_step != gt_step:
                    break

                step_idx += 1
                step_save_dir = ensure_dir(os.path.join(task_save_dir, f"step_{step_idx}"))
                print(action_step)
                flog.write(action_step + "\n")

                target_link_name, manip_way = action_step.split(": ")
                link_type = link_dict.get(target_link_name, "")

                if "button" in link_type:
                    joint_angles, success_flag = push_button(
                        flog,
                        env,
                        args,
                        step_save_dir,
                        shape_id,
                        target_link_name,
                        link_type,
                        joint_angles,
                        view,
                        img_size,
                    )
                elif "knob" in link_type:
                    match = re.search(r"(\d+)", manip_way)
                    if not match:
                        success_flag = False
                    else:
                        rotation_degrees = int(match.group(1))
                        joint_angles, success_flag = rotate_knob(
                            flog,
                            env,
                            args,
                            step_save_dir,
                            shape_id,
                            target_link_name,
                            link_type,
                            rotation_degrees,
                            joint_angles,
                            view,
                            img_size,
                        )
                elif any(keyword in link_type for keyword in ["lid", "slider", "screen", "drawer", "door"]):
                    joint_angles, success_flag = manip_slider_or_door(
                        flog,
                        env,
                        args,
                        step_save_dir,
                        manual_subdir,
                        shape_id,
                        target_link_name,
                        link_type,
                        joint_angles,
                        joint2world_dict,
                        view,
                        img_size,
                        use_cached_grasp=not args.no_cache_grasp,
                    )
                else:
                    flog.write("No primitive action\n")
                    success_flag = False

                if not success_flag:
                    flog.write(f"Failed at step {step_idx}\n")
                    break
                success_step_count += 1

            completion_rate = success_step_count / len(gt_steps) if gt_steps else 0.0
            flog.write(f"'{task_name}' Completion Rate: {completion_rate}\n")
            result_dict[source_sample_dir]["completion_rates"].append(completion_rate)

            if gt_steps == planned_steps and success_flag:
                success_execution += 1
                result_dict[source_sample_dir]["success_task_execution"].append(task_name)
                flog.write(
                    f"Successfully complete task '{task_name}'! Success Execution Rate: "
                    f"{success_execution / total_planning}\n"
                )

            flog.write("#########################################################\n")
            print("---------------")

        flog.write("----------------------------------------------------------------\n\n")
        return result_dict
    finally:
        flog.close()
        env.close()


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
    parser.add_argument("--foundationpose_url", default="http://127.0.0.1:6006/foundationpose_flask", type=str)
    parser.add_argument("--grasp_server_url", default=None, type=str)
    parser.add_argument("--global_tasks_path", default=None, type=str)
    parser.add_argument("--global_tasks_key", default=None, type=str)
    parser.add_argument("--sample", default=None, type=str)
    parser.add_argument("--max_samples", default=None, type=int)
    parser.add_argument("--max_tasks", default=None, type=int)
    parser.add_argument("--obj_init_scale", default=0.4, type=float)
    parser.add_argument("--llm_version", default="gpt-4o", type=str)
    parser.add_argument("--vlm_version", default="gpt-4o", type=str)
    parser.add_argument("--gui", action="store_true", default=False)
    parser.add_argument("--no_cache_pose", action="store_true", default=False)
    parser.add_argument("--no_cache_plan", action="store_true", default=False)
    parser.add_argument("--no_cache_grasp", action="store_true", default=False)
    parser.add_argument("--use_gt_alignment", action="store_true", default=False)
    parser.add_argument("--use_gt_plan", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.out_dir = normalize_out_dir(args.out_dir)
    ensure_dir(args.out_dir)
    args.runtime_cache_dir = ensure_dir(os.path.join(args.out_dir, "runtime_cache"))
    args.log_path = normalize_results_file(
        args.log_path,
        os.path.join(args.out_dir, "track2.log"),
    )
    args.result_path = normalize_results_file(
        args.result_path,
        os.path.join(args.out_dir, "track2_results.json"),
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
        source_sample_dir = get_source_sample_dir(args, manual_subdir)
        if source_sample_dir in result_dict:
            continue
        shape_id = manual_subdir.split("_")[0]
        pred_obj_pose = estimate_pose(args, shape_id, manual_subdir)
        result_dict = benchmark(
            args,
            manual_subdir,
            result_dict,
            pred_obj_pose,
            obj_init_scale=args.obj_init_scale,
        )
        with open(args.result_path, "w", encoding="utf-8") as file:
            json.dump(result_dict, file, ensure_ascii=False, indent=4)

    if total_planning > 0:
        print(f"Track 2 planning SR: {success_planning / total_planning:.4f}")
    if total_planning > 0:
        print(f"Track 2 execution SR: {success_execution / total_planning:.4f}")
    
