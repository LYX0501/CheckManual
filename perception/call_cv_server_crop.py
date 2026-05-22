import requests
import argparse
import os

def call_cv_server(img_path, category):
    port_num = os.environ.get("CHECKMANUAL_CV_SERVER_PORT", "5001").strip()
    cv_server_url = f'http://localhost:{port_num}/crop_appliance'
    data = {"img_path": img_path, "category": category}
    print(data)
    response = requests.post(cv_server_url, json=data)
    print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send image path to CV server.')
    parser.add_argument('img_path', type=str, help='Path to the image')
    parser.add_argument('category', type=str, help='appliance category')
    args = parser.parse_args()
    
    call_cv_server(args.img_path, args.category)
