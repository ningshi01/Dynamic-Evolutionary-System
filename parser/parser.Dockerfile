# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY parser-requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir --timeout=600 -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Copy the content of the local src directory to the working directory
COPY parser-main.py main.py

# Expose the port the app runs on
EXPOSE 8000

# Run the command to start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]