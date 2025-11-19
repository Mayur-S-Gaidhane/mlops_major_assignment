FROM python:3.9-slim

WORKDIR /app

COPY requirements.docker.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.docker.txt

# copy source
COPY src /app/src
COPY artifacts /app/artifacts

# default command - run predict (CI will run this to verify)
CMD ["python", "src/predict.py"]
