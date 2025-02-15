FROM python:3.13-slim-bookworm

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD [ "streamlit","run", "./analisisItems.py" ]