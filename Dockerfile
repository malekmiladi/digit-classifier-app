FROM python:3.11.6-alpine
COPY ./web-app/ /opt/digit-classifier/
RUN  pip install -r /opt/digit-classifier/requirements.txt
WORKDIR /opt/digit-classifier
CMD ["python3", "-m", "flask", "--app", "server", "run", "-h", "0.0.0.0", "-p", "8080"]
