import os
import re
import shutil
from api_utils.gpt_api import *
from api_utils.ocr_api import *

from pdf2image import convert_from_path
from PIL import Image

def resize_image(image_path, size=(1600, 2400)):
    with open(image_path, "rb") as image_file:
        image = Image.open(image_file)
        image = image.resize(size, Image.ANTIALIAS)
    
    image.save("calling_gpt.png")
    
    return "calling_gpt.png"

def sort_png(folder_path):
    pattern = re.compile(r'\d+')
    files_with_numbers = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            match = pattern.search(filename)
            if match:
                number = int(match.group())
                files_with_numbers.append((filename, number))
                
    sorted_files = sorted(files_with_numbers, key=lambda x: x[1])
    return [file[0] for file in sorted_files]

def convert_pdf_to_png(pdf_path, output_folder):
    images = convert_from_path(pdf_path)
    for i, page in enumerate(images):
        page.save(f'{output_folder}/{i + 1}.png', 'PNG')
        
    return len(images)

def get_manual_ocr(manual_dir):
    manual_ocr_path = os.path.join(manual_dir, "manual_ocr_result.json")
    print("Call OCR API")
    manual_imgs = sort_png(manual_dir)
    print("Total page number: ", len(manual_imgs))
    
    manual_ocr_dict = {}
    for manual_img in manual_imgs:
        manual_img_path = os.path.join(manual_dir, manual_img)
        ocr_result = ocr_detection(manual_img_path)
        manual_ocr_dict[manual_img] = ocr_result
        with open(manual_ocr_path, 'w', encoding='utf-8') as file:
            json.dump(manual_ocr_dict, file, ensure_ascii=False, indent=4)
        
    return manual_ocr_dict

def conv_manual_content(manual_ocr_dict):
    manual_content = ""
    for page_name, ocr_result in manual_ocr_dict.items():
        page_idx = page_name.replace(".png","")
        page_content = " | ".join([item["words"] for item in ocr_result])
        manual_content += f"Page {page_idx}：{page_content}\n"
    
    return manual_content