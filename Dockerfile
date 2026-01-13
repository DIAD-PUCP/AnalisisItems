FROM python:3.14-slim-trixie

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD [ "streamlit","run", "./analisisItems.py" ]
