FROM python:3.9-slim

WORKDIR /app

COPY . /app

# RUN pip3 install -r requirement.txt 
RUN pip3 install scikit-learn==1.1.3 gunicorn flask flask-cors

CMD gunicorn --bind 0.0.0.0:5005 --timeout=150 app:app -w 1

EXPOSE 5005