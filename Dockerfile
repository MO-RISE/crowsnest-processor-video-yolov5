FROM python:3.9-slim

RUN apt-get update && apt-get install --no-install-recommends -y \
    ffmpeg\
    libsm6\
    libxext6\
    git\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY main.py main.py

CMD ["python3", "main.py"]