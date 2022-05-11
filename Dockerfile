FROM python:3.9-slim

RUN apt-get update && apt-get install -y python3-opencv

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

WORKDIR /app

COPY main.py main.py

CMD ["python3", "main.py"]