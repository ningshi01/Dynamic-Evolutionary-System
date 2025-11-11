# Minimal image for the request simulator
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY simulate_requirements.txt requirements.txt
RUN pip install --no-cache-dir --timeout=600  -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Add simulator script (provide config via volume or bake it in a derived image)
COPY request_simulator.py main.py

# Document required runtime envs (set at `docker run` time)
#   SIM_URL, SIM_RPS, SIM_DURATION, SIM_TIMEOUT
# Optional:
#   SIM_CONFIG_FILE -> /app/request_simulator.config.json
EXPOSE 8000

CMD ["python", "main.py"]
