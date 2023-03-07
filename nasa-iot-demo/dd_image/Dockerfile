FROM gcr.io/mwpmltr/nasa-iot-base:v1
RUN pip install -U data-describe
WORKDIR /app
COPY dd.py .

ENTRYPOINT ["python", "dd.py"]
