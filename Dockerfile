FROM python:3.10.13
RUN pip install --upgrade pip
RUN apt-get update && apt-get -y install enchant-2
COPY ./app/requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install dash[diskcache]
RUN pip install transformers[sentencepiece]
COPY ./app /app
EXPOSE 8080
ENTRYPOINT ["python", "app.py"]
