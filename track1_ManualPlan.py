import os
import re
import cv2
import json
import shutil
import numpy as np
from env import Env
from camera import Camera
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from pdf_utils import convert_pdf_to_png, get_manual_ocr, conv_manual_content
from api_utils.gpt_api import encode_image, gptv_response, gpt_response

total_alignment, success_alignment = 0, 0
total_planning, success_planning = 0, 0

def get_camera_angles(view):
    if view == "front":
        return np.pi / 15, np.pi
    elif view == "top":
        return np.pi / 2, np.pi
    elif view == "left":
        return np.pi / 15, np.pi * 0.5
    elif view == "right":
        return np.pi / 15, np.pi * 1.5
    elif view == "left-front":
        return np.pi / 15, np.pi * 0.9
    elif view == "right-front":
        return np.pi / 15, np.pi * 1.2
    elif view == "back":
        return np.pi / 15, np.pi * 1.999
    else:
        raise ValueError("Invalid view specified")

def extract_gt_link_function(part_state_dict):
    gt_link_function_dict = {}
    pattern = re.compile(r'(link_\d+)\s+([A-Za-z\s]+)\s+\(')
    
    for string in part_state_dict.keys():
        match = pattern.match(string)
        if match:
            key = match.group(1)
            value = match.group(2).strip()
            gt_link_function_dict[key] = value

    return gt_link_function_dict


def resolve_manual(manual_dir_path, manual_file_name):
    manual_pngs_dir = os.path.join(manual_dir_path, "manual_pngs")
    ocr_json_path = os.path.join(manual_pngs_dir, "manual_ocr_result.json")
    
    if os.path.exists(manual_pngs_dir) and not os.path.exists(ocr_json_path):
        shutil.rmtree(manual_pngs_dir)
    
    if os.path.exists(ocr_json_path):
        print("Loaded manual_ocr_result.json")
        with open(ocr_json_path, 'r', encoding='utf-8') as file:
            manual_ocr_dict = json.load(file)
    else:
        os.makedirs(manual_pngs_dir, exist_ok=True)
        manual_pdf_path = os.path.join(manual_dir_path, manual_file_name)
        convert_pdf_to_png(manual_pdf_path, manual_pngs_dir)
        manual_ocr_dict = get_manual_ocr(manual_pngs_dir)
    
    manual_content = conv_manual_content(manual_ocr_dict)
    return manual_pngs_dir, manual_ocr_dict, manual_content

def vis_analyze_page(manual_pngs_dir, vlm_version="gpt-4o"):
    manual_vis_info_path = os.path.join(manual_pngs_dir, f"manual_vis_info_{vlm_version}.json")
    if os.path.exists(manual_vis_info_path):
        print("Loaded manual_vis_info.json")
        with open(manual_vis_info_path, 'r', encoding='utf-8') as file:
            page_vis_info_dict = json.load(file)
        return page_vis_info_dict
        
    manual_pngs = [file for file in os.listdir(manual_pngs_dir) if file.endswith(".png")]
    page_vis_info_dict = dict()
    for page_idx in range(len(manual_pngs)):
        manual_png = f"{page_idx+1}.png"
        manual_png_path = os.path.join(manual_pngs_dir, manual_png)
        encoded_img = encode_image(manual_png_path)
        
        text_prompt = "This is a page from an appliance manual. Please describe which figures and tables you can see on this page, \
            and what information is contained in each firgure and table respectively."
        prompt_content = [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}", "detail": "high"}}
        ]
        response = gptv_response([{"role": "user", "content": prompt_content}], model_version=vlm_version)  
        print(manual_png, response)
        page_vis_info_dict[manual_png] = response

    with open(manual_vis_info_path, 'w', encoding='utf-8') as file:
        json.dump(page_vis_info_dict, file, ensure_ascii=False, indent=4)
    
    return page_vis_info_dict

def capture_link_photos(env, manual_dir_path, link_nameid_dict, img_size):
    link_pngs_dir = os.path.join(manual_dir_path, "link_pngs")
    if os.path.exists(link_pngs_dir):
        link_pngs = [f for f in os.listdir(link_pngs_dir) if f.endswith(".png")]
        if len(link_pngs) == len(link_nameid_dict.keys()):
            return link_pngs_dir
        else:
            shutil.rmtree(link_pngs_dir)
    
    os.mkdir(link_pngs_dir)

    candidate_views = ["front", "left", "right", "left-front", "right-front", "top", "back"]
    
    for link_name, link_id in link_nameid_dict.items():
        link_mask_list, link_mask_area_list, rgb_list = [], [], []
        for view in candidate_views:
            phi, theta = get_camera_angles(view)
            cam = Camera(env, image_size=img_size, dist=3.0, phi=phi, theta=theta, fixed_position=True)
            
            env.step()
            env.render()
            rgb, depth = cam.get_observation()
            marked_rgb = (rgb * 255).astype(np.uint8)
            
            link_mask = cam.get_movable_link_mask([link_id])
            link_mask_area = np.sum(link_mask)
            link_mask_area_list.append(link_mask_area)
            link_mask_list.append(link_mask)
            rgb_list.append(marked_rgb)
        
        selected_view_idx = np.argmax(link_mask_area_list)
        selected_view = candidate_views[selected_view_idx]
        selected_view_link_mask = link_mask_list[selected_view_idx]
        selected_view_rgb = rgb_list[selected_view_idx]
        
        
        plt.figure(figsize=(40, 40))
        plt.imshow(selected_view_rgb)
        img = np.ones((selected_view_link_mask.shape[0], selected_view_link_mask.shape[1], 4))
        img[:, :, 3] = 0
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(selected_view_link_mask.astype(np.uint8), kernel, iterations=3)
        outline_mask = dilated_mask - selected_view_link_mask
        img[outline_mask > 0] = [1, 0, 0, 1]  # Set outline pixels to red
        plt.imshow(img)
        plt.axis('off')
        save_path = os.path.join(link_pngs_dir, f"{link_name}_mask.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    return link_pngs_dir

def align_parts(manual_dir_path, link_pngs_dir, page_vis_info_dict, llm_version="gpt-4o", vlm_version="gpt-4o"):
    link_function_json = os.path.join(manual_dir_path, "link_pngs", f"pred_link_function_{llm_version}.json")
    if os.path.exists(link_function_json):
        with open(link_function_json, 'r', encoding='utf-8') as file:
            link_function_dict = json.load(file)
        return link_function_dict
    
    page_vis_info = ""
    for page_file, page_content in page_vis_info_dict.items():
        page_idx = page_file.replace(".png", "")
        page_vis_info += f"Page {page_idx}: {page_content} | \n"
    
    prompt = [
        {"role": "system", "content": "I have leveraged multimodal large model to analyze each page of an appliance manual. \
                Please tell me which pages display figures about introducing appliance components overview and control panel. Your answer should be a dictionary, \
                in which each item takes page number as key and corresponding figure information as value. You can ignore the first page."},
        {"role": "user", "content": f"Appliance Manual Content:\n{page_vis_info}"}
    ]
    while True:
        response = gpt_response(prompt, model_version=llm_version)
        print("raw pred: ", response)

        key_value_pattern = re.compile(r'(\d+):\s*"([^"]+)"')
        matches = key_value_pattern.findall(response)
        resolved_dict = {int(key): value for key, value in matches}
        print("resolved dict: ", resolved_dict)
        
        figure_page_idxs = sorted(resolved_dict.keys())[:3]
        if len(figure_page_idxs) > 0:
            break
    figure_pages = [os.path.join(manual_dir_path, "manual_pngs", f"{k}.png") for k in figure_page_idxs]
    
    link_function_dict = {}
    link_pngs = os.listdir(link_pngs_dir)
    for link_png in link_pngs:
        link_name = link_png.replace("_mask.png","")
        link_png_path = os.path.join(link_pngs_dir, link_png)
        encoded_link_img = encode_image(link_png_path)
        
        text_prompt = f"I will provide you with an appliance photo and several page screenshots from the appliance manual. \
            On the appliance photo, the target component is surrounded by red lines. This target component is labeled with function name in the manual. \
            Please analyze every diagrams in the given manual pages and answer the function name of this target component. \
            Your answer should be a dictionary which takes '{link_name}' as key and the function name as value. \
            Already aligned link ids and function names are {link_function_dict}, whose function names should be avoided."
        prompt_content = [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_link_img}", "detail": "high"}}
        ]  
        while True:
            copied_prompt_content = prompt_content.copy()
            for figure_page in figure_pages:
                encoded_page_img = encode_image(figure_page)
                copied_prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_page_img}", "detail": "high"}})
            response = gptv_response([{"role": "user", "content": copied_prompt_content}], model_version=vlm_version)  
            response = response.replace("\"", "'")
            link_function_pattern = r"'(\w+)' *: *'([^']*)'"
            matches = re.findall(link_function_pattern, response)
            link_function_alignment = {key: value for key, value in matches}
            if len(link_function_alignment) == 1:
                break
            print("Retry")
            
        link_function = link_function_alignment[link_name]
        link_function_dict[link_name] = link_function
        print(f"Align ", link_function_alignment)
    
    with open(link_function_json, 'w', encoding='utf-8') as file:
        json.dump(link_function_dict, file, ensure_ascii=False, indent=4)
    print(f"Saved {link_function_json}")
    
    return link_function_dict

def plan_steps(manual_dir_path, task_idx, manual_content, task_name, pred_link_function_dict, llm_version="gpt-4o", use_cache_plan=True):
    task_save_path = os.path.join(manual_dir_path, f"task_{task_idx}_ManualPlan_{llm_version}_plans.json")
    
    if os.path.exists(task_save_path) and use_cache_plan:
        with open(task_save_path, 'r', encoding='utf-8') as file:
            task_plan = json.load(file)
        return task_plan["task_plan"]  
    else:
        prompt = [
            {"role": "system", "content": "I will provide you with an appliance manual and a dictionary about mapping relation between link id and component name. \
                Please help me plan how to complete the given task. If the task is metioned in the manual, you can follow the manual content. \
                Otherwise, you should combine common sense and appliance part functions to make plan. Every planing step should follow 'link_id: Operation' like 'link_1: Press 1 time'. \
                Candidate operations only include ['Press ... times', 'Rotate ... degrees', 'Open', 'Close']. Your answer must be a python list sequenced from step 1 to step N."},
            {"role": "user", "content": f"Task: {task_name} ; Appliance Manual:\n{manual_content} ; Link ID - Component Name Relation: {pred_link_function_dict}"}
        ]
        while True:
            response = gpt_response(prompt, model_version=llm_version)
            response = response.replace("\"","'").replace("link ","link_")
            print(response)
        
            matches = re.findall(r"'(link_\d+: [^']+)'", response)
            if len(matches) > 0:
                break
            
        planned_steps = [match for match in matches]
        
        task_dict = {f"task_plan": planned_steps}
        with open(task_save_path, 'w', encoding='utf-8') as file:
            json.dump(task_dict, file, ensure_ascii=False, indent=4)
        
        print(f"Saved {task_save_path}")
    
    return planned_steps


def eval_track1(manual_dir_path, result_dict, view="front", img_size=1024):
    global total_alignment, success_alignment, total_planning, success_planning
    
    manual_file_name = [f for f in os.listdir(manual_dir_path) if f.lower().endswith('.pdf')][0]
    manual_id = manual_file_name.replace(".pdf","")
    print(f"Start to eval {manual_dir_path}/{manual_file_name}")
    
    asset_id = manual_file_name.split("_")[0]
    
    result_dict[manual_id] = {"total_tasks": [], "success_task_plan": [], "alignment": 0}
    
    # Open task file
    task_file_name = "eval_tasks.json"
    with open(os.path.join(manual_dir_path, task_file_name), 'r') as file:
        task_dict = json.load(file)

    # Open part state function file
    part_state_functions_file_name = "part_state_functions.json"
    with open(os.path.join(manual_dir_path, part_state_functions_file_name), 'r') as file:
        part_state_functions = json.load(file)
    
    gt_link_function_dict = extract_gt_link_function(part_state_functions)
    
    # Setup environment
    env = Env()
    asset_urdf_fn = 'data/sapien_dataset/%s/mobility.urdf' % asset_id
    asset_material = env.get_material(4, 4, 0.01)
    joint_angles = env.load_object(asset_urdf_fn, asset_material, rotation=view, state="zero")
    link_nameid_dict = {name:id for name, id in zip(env.movable_link_names, env.movable_link_ids) if name in gt_link_function_dict.keys()}
    
    # Create camera
    phi, theta = get_camera_angles(view)
    cam = Camera(env, image_size=img_size, dist=5.0, phi=phi, theta=theta, fixed_position=True)
    
    # Resolve manual
    manual_pngs_dir, manual_ocr_dict, manual_content = resolve_manual(manual_dir_path, manual_file_name)
    page_vis_info_dict = vis_analyze_page(manual_pngs_dir)
    
    # Align parts
    link_pngs_dir = capture_link_photos(env, manual_dir_path, link_nameid_dict, img_size)
    pred_link_function_dict = align_parts(manual_dir_path, link_pngs_dir, page_vis_info_dict)

    total_alignment += 1
    if gt_link_function_dict == pred_link_function_dict:
        success_alignment += 1
        result_dict[manual_id]["alignment"] = 1
    align_sr = success_alignment / total_alignment
    print("Current Alignment Success Rate:", align_sr)

    # Eval
    task_idx = 0
    for task_name, gt_steps in task_dict.items():
        task_idx += 1
        print("Task Name: ", task_name)
        gt_steps = [item[1].split()[0]+": "+item[2] for item in gt_steps]
        planned_steps = plan_steps(manual_dir_path, task_idx, manual_content, task_name, pred_link_function_dict, use_cache_plan=False)
        
        total_planning += 1
        result_dict[manual_id]["total_tasks"].append(task_name)
        if gt_steps == planned_steps:
            success_planning += 1
            result_dict[manual_id]["success_task_plan"].append(task_name)
        plan_sr = success_planning / total_planning
        print("Current Plan Success Rate:", plan_sr)

    env.close()
    print("------------------------------------------------------\n")
    
    return result_dict

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--manual_data_path', default='data/CheckManual_Data', type=str)
    parser.add_argument('--track1_result_path', default='track1_result.json', type=str)
    args = parser.parse_args()
    
    manual_dirs = os.listdir(args.manual_data_path)
    sorted_manual_dirs = sorted(
        manual_dirs,
        key=lambda x: int(x.split('_')[-1]) if x.startswith('manual_') and x.split('_')[-1].isdigit() else float('inf')
    )
    
    if os.path.exists(args.track1_result_path):
        with open(args.track1_result_path, 'r', encoding='utf-8') as file:
            result_dict = json.load(file)
    else:
        result_dict = dict()
    
    for manual_dir in sorted_manual_dirs:
        manual_dir_path = os.path.join(args.manual_data_path, manual_dir)
        manual_file_name = [f for f in os.listdir(manual_dir_path) if f.lower().endswith('.pdf')][0]
        manual_id = manual_file_name.replace(".pdf","")
        if manual_id in result_dict.keys():
            continue
        
        result_dict = eval_track1(manual_dir_path, result_dict)
        with open(args.track1_result_path, 'w', encoding='utf-8') as file:
            json.dump(result_dict, file, ensure_ascii=False, indent=4)
        
    # Result Statistics
    category_result_dict = {}
    for manual_id, results in result_dict.items():
        category = manual_id.split("_manual")[0]
        category = "_".join(category.split("_")[1:])
        
        if category not in category_result_dict.keys():
            category_result_dict[category] = {"total_tasks": 0, "success_task_plan": 0, "total_alignment": 0, "success_alignment": 0}
        
        category_result_dict[category]["total_alignment"] += 1
        category_result_dict[category]["success_alignment"] += results["alignment"]
        category_result_dict[category]["total_tasks"] += len(results["total_tasks"])
        category_result_dict[category]["success_task_plan"] += len(results["success_task_plan"])
    
        for category, results in category_result_dict.items():
            align_sr = results["success_alignment"] / results["total_alignment"]
            plan_sr = results["success_task_plan"] / results["total_tasks"]
            print(f"{category} | Alignment SR: {align_sr};  Plan SR: {plan_sr}")