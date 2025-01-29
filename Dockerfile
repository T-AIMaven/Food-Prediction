FROM python:3.10.6-buster

WORKDIR /prod

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python

COPY foodscore foodscore
COPY raw_data raw_data
COPY setup.py setup.py
RUN pip install .


CMD uvicorn foodscore.api.fast:app --host 0.0.0.0 --port $PORT
