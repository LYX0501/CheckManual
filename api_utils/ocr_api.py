import json
import requests
import base64

with open("api_utils/api_key_config.json", 'r', encoding='utf-8') as file:
    api_key_config = json.load(file)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def get_access_token():
    API_KEY = api_key_config["OCR"]["API_KEY"]
    SECRET_KEY = api_key_config["OCR"]["SECRET_KEY"]
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

def ocr_detection(img_path):
    encoded_image = encode_image(img_path)
    access_token = get_access_token()
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
    # request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate"
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    for _ in range(100):
        try:
            ocr_response = requests.post(request_url, data={"image": encoded_image}, headers=headers).json()["words_result"]
            print(ocr_response)
            break
        except KeyboardInterrupt:
            print("Stop")
            break
        except Exception as e:
            print("RETRY OCR API:", e)
            continue
    
    return ocr_response
