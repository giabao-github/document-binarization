FROM python:3.10-slim
WORKDIR /app
COPY requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY . /app
CMD ["python", "scripts/binarize.py"]
