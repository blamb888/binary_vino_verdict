FROM --platform=linux/amd64 python:3.10.6-slim

COPY api /api
COPY requirements_api.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT