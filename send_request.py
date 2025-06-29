import requests
import json
import os

# 构建文件的绝对路径
file_path = os.path.join(os.path.dirname(__file__), 'add_docs_payload.json')

# 从文件加载JSON数据
with open(file_path, 'r') as f:
    payload = json.load(f)

# 设置请求头
headers = {
    "Content-Type": "application/json"
}

# 设置URL
url = "http://192.168.5.76:32319/add_documents"

try:
    # 发送POST请求
    response = requests.post(url, headers=headers, json=payload)

    # 检查响应状态码
    response.raise_for_status() 

    # 打印成功的响应内容
    print("请求成功!")
    print("响应内容:", response.json())

except requests.exceptions.HTTPError as errh:
    print(f"Http Error: {errh}")
    print(f"Response body: {response.text}")
except requests.exceptions.ConnectionError as errc:
    print(f"Error Connecting: {errc}")
except requests.exceptions.Timeout as errt:
    print(f"Timeout Error: {errt}")
except requests.exceptions.RequestException as err:
    print(f"Oops: Something Else: {err}")
    print(f"Response body: {response.text}") 