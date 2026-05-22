import argparse
import os
import pickle
import re
from flask import Flask, jsonify, request
from cv_utils.groundingdino_tool import *
from cv_utils.sam_tool import *
from cv_utils.draw_tool import *

gdino_model = initialize_dino_model()
sam_prompt_predictor, sam_general_predictor = initialize_sam_model()

app = Flask(__name__)

def crop_obj4sam(img_path, category, resized_height=800):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    appliance_candidate_masks = []
    gdino_detections = None
    for target_label in ["appliance", category, "equipment", "object"]:
        try:
            gdino_detections = openset_detection(image, target_label, gdino_model)
            appliance_candidate_masks = sam_prompt_masking(
                image,
                gdino_detections.xyxy,
                sam_prompt_predictor,
            )
        except Exception as exc:
            print(f"GroundingDINO crop failed for '{target_label}': {type(exc).__name__}: {exc}")
            appliance_candidate_masks = []
            gdino_detections = None
        if appliance_candidate_masks:
            break

    if not appliance_candidate_masks or gdino_detections is None or len(gdino_detections.xyxy) == 0:
        height, width = image.shape[:2]
        return image, width, height

    appliance_mask = appliance_candidate_masks[0]
    image[~appliance_mask] = [255, 255, 255] 
    crop_coords = gdino_detections.xyxy[0]
    
    width, height = int(crop_coords[2]-crop_coords[0]), int(crop_coords[3]-crop_coords[1])
    resized_width = int((resized_height/height)*width)
    
    cropped_image = image[int(crop_coords[1]):int(crop_coords[3]), int(crop_coords[0]):int(crop_coords[2])]
    resized_image = cv2.resize(cropped_image, (resized_width, resized_height))
    
    return resized_image, resized_width, resized_height

def sam_general_masks(image, save_path):
    masks = sam_general_predictor.generate(image)

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight',pad_inches=0)
    plt.close()
    
    return masks


@app.route('/sam', methods=['POST'])
def sam():
    data = request.get_json()
    img_path = data.get('img_path')
    category = data.get('category')

    croped_appliance, width, height = crop_obj4sam(img_path, category)
    # croped_appliance = cv2.imread(img_path)
    croped_appliance = cv2.cvtColor(croped_appliance, cv2.COLOR_BGR2RGB)
    height, width, channels = croped_appliance.shape
    
    cropped_rgb_path = img_path.replace("rgb", "cropped_rgb")
    cv2.imwrite(cropped_rgb_path, croped_appliance)
    
    cropped_rgb_masked_path = img_path.replace("rgb", "cropped_rgb_masked")
    masks = sam_general_masks(croped_appliance, cropped_rgb_masked_path)
    masks_path = img_path.replace(".png", ".pkl").replace("rgb", "all_masks")
    with open(masks_path, 'wb') as f:
        pickle.dump(masks, f)
    
    cropped_rgb_masked_ids_path = img_path.replace("rgb", "cropped_rgb_masked_ids")
    draw_idx(cropped_rgb_masked_path, masks, width, height, save_path=cropped_rgb_masked_ids_path)
    
    
    return jsonify({'state': "success"})


@app.route('/crop_appliance', methods=['POST'])
def crop_appliance():
    data = request.get_json()
    img_path = data.get('img_path')
    category = data.get('category')

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for target_label in ["appliance", category, "equipment", "object"]:
        gdino_detections = openset_detection(image, target_label, gdino_model)
        appliance_candidate_masks = sam_prompt_masking(image, gdino_detections.xyxy, sam_prompt_predictor)
        if len(appliance_candidate_masks) > 0:
            break
    
    appliance_mask = appliance_candidate_masks[0]
    
    obj_mask_path = img_path.replace(".png", ".npy").replace("rgb", "obj_mask")
    np.save(obj_mask_path, appliance_mask)
    
    return jsonify({'state': "success"})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)
