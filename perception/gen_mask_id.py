import os
import re
import pickle
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
    
    for target_label in ["appliance", category, "equipment", "object"]:
        gdino_detections = openset_detection(image, target_label, gdino_model)
        appliance_candidate_masks = sam_prompt_masking(image, gdino_detections.xyxy, sam_prompt_predictor)
        if len(appliance_candidate_masks) > 0:
            break
    
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



img_path = "/vepfs-cnsh4137610c2f4c/algo/user9/Manip/data/real_microwave/rgb.jpg"
category = "real_microwave"

# img_path = "/vepfs-cnsh4137610c2f4c/algo/user9/manual_eval_group1/101947_oven_1/rgb_track2.png"
# category = "oven"

# img_path = "/vepfs-cnsh4137610c2f4c/algo/user9/manual_eval_group1/103361_washing_machine/rgb_track2.png"
# category = "washing machine"

# img_path = "/vepfs-cnsh4137610c2f4c/algo/user9/manual_eval_group1/7119_microwave/rgb_track2.png"
# category = "microwave"
    
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
