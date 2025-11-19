FROM python:3.9-slim

WORKDIR /app

COPY requirements.docker.txt /app/requirements.docker.txt
RUN pip install --no-cache-dir -r /app/requirements.docker.txt

COPY src /app/src
COPY artifacts /app/artifacts

CMD ["python", "src/predict.py"]
