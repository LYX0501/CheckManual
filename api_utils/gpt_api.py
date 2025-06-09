import json
import requests
import base64

with open("api_utils/api_key_config.json", 'r', encoding='utf-8') as file:
    api_key_config = json.load(file)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def gpt_response(prompt, model_version='gpt-4o'):
    for idx in range(100):
        try:
            # print("read GPT key")
            key = api_key_config["GPT"]["key"] 
            url = api_key_config["GPT"]["url"]
            header = {
                "Content-Type":"application/json",
                "Authorization": key
            }
            post_dict = {
                "model": model_version,
                "messages": prompt,
            }
            print("Try request GPT")
            r = requests.post(url, json=post_dict, headers=header)
            json_r = r.json()
            print(json_r["choices"][0]["message"])
            message = json_r["choices"][0]["message"]["content"]
            break
        except KeyboardInterrupt:
            print("Stop")
            break
        except Exception as e:
            print(e)
            # print(r.json())
            pass
    # price = r['usage']['completion_tokens']/1000*0.43 + r['usage']['prompt_tokens']/1000*0.22
    # print("gpt4 used time: %.2f, used price: %.5f"%(time.time()-start_time,price))
    return message

def gptv_response(prompt, model_version='gpt-4o'):
    for idx in range(100):
        try:
            key = api_key_config["GPT"]["key"] 
            url = api_key_config["GPT"]["url"]
            header = {
                "Content-Type":"application/json",
                "Authorization": key
            }
            post_dict = {
                "model": model_version,
                "messages": prompt,
            }
            r = requests.post(url, json=post_dict, headers=header)
            json_r = r.json()
            if "choices" not in json_r.keys():
                print(json_r)
            message = json_r["choices"][0]["message"]["content"]
            break
        except KeyboardInterrupt:
            print("Stop")
            break
        except Exception as e:
            print(e)
            # print(json_r)
            # print(r.json())
            continue
    return message