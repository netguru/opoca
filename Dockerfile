FROM python:3.7-slim-buster

ADD requirements.txt /home/
ADD opoca /home/

WORKDIR /home

RUN pip install -r requirements.txt
