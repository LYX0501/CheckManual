import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置
import matplotlib.pyplot as plt
import os

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DEFAULT_CACHE_DIR = os.path.join(ROOT_DIR, "results", "cache")

def draw_bbox(raw_real_image, interaction, masks, save_path=None):
    if save_path is None:
        save_path = os.path.join(DEFAULT_CACHE_DIR, "bbox.jpg")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    raw_image = cv2.imread(raw_real_image)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    part_mask = {}
    for item in interaction:
        idx, name = item.split(":")
        mask_idx = int(idx)-1
        part_mask[name] = {'masks': masks[mask_idx]}
        
        x, y, w, h = masks[mask_idx]["bbox"]
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(raw_image, top_left, bottom_right, (0, 0, 255), 2)
        (text_width, text_height), baseline = cv2.getTextSize(str(idx), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(raw_image, idx, (top_left[0]+5, top_left[1] + text_height+8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(save_path, raw_image)
    
def draw_idx(image_path, masks, width, height, threshold=1000, save_path=None):
    if save_path is None:
        save_path = os.path.join(DEFAULT_CACHE_DIR, "crop_masked_bbox.jpg")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    text_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    
    # for idx, mask in enumerate(masks):
    #     x, y, w, h = mask["bbox"]
    #     if w*h > threshold > mask["predicted_iou"] > 0.97:
    #         top_left = (x, y)
    #         bottom_right = (x + w, y + h)
    #         cv2.rectangle(image_rgb, top_left, bottom_right, (0, 0, 255), 2)
    
    for idx, mask in enumerate(masks):
        x, y, w, h = mask["bbox"]
        # if w*h > threshold and mask["predicted_iou"] > 0.97:
        if mask["predicted_iou"] > 0.97:
            print(idx, mask["predicted_iou"])
            # top_left = (x, y)
            # bottom_right = (x + w, y + h)
            # (text_width, text_height), baseline = cv2.getTextSize(str(idx+1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # text_area = (x-15, y-text_height, text_width, text_height)
            # if np.any(text_mask[text_area[1]:text_area[1]+text_area[3], text_area[0]:text_area[0]+text_area[2]]):
            #     continue
            # else:
            #     cv2.putText(image_rgb, str(idx+1), (top_left[0]+5, top_left[1] + text_height+8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            #     text_mask[text_area[1]:text_area[1]+text_area[3], text_area[0]:text_area[0]+text_area[2]] = 255
            
            contours, _= cv2.findContours(mask["segmentation"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            contour = max(contours, key = cv2.contourArea)
            M = cv2.moments(contour)
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            (text_width, text_height), baseline = cv2.getTextSize(str(idx+1), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            text_area = (cx-15, cy-text_height, text_width, text_height)
            if np.any(text_mask[text_area[1]:text_area[1]+text_area[3], text_area[0]:text_area[0]+text_area[2]]):
                # print("Text Overlap")
                continue
            else:
                top_left = (cx-15, cy-text_height)
                bottom_right = (cx-15+text_width, cy)
                image_rgb = cv2.rectangle(image_rgb, top_left, bottom_right, (0, 0, 0), -1)
                cv2.putText(image_rgb, str(idx+1), (cx-15, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                text_mask[text_area[1]:text_area[1]+text_area[3], text_area[0]:text_area[0]+text_area[2]] = 255


    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image_bgr)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.3 ]])
        img[m] = color_mask
    ax.imshow(img) 
