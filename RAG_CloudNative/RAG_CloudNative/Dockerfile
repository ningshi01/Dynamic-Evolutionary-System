# 使用轻量级 Python 基础镜像
FROM python:3.9-slim

# 设置容器内的工作目录
WORKDIR /app

# 先复制依赖文件（这样能利用 Docker 缓存，加快构建）
# 如果没有 requirements.txt，也不会报错
COPY fileservice-requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# 再复制项目代码（包括 .py 和 .yaml）
COPY . /app

# 暴露服务端口（默认 8000，可改）
EXPOSE 8000

# 默认启动命令（假设入口是 main.py 里定义的 app 对象）
# 如果你用 Flask 或者不同的文件名，可以改 CMD
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
