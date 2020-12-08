FROM gcr.io/mwpmltr/rrusson_kubeflow_base:v1
RUN pip install -U data-describe
WORKDIR /app
COPY dd.py .

ENTRYPOINT ["python", "dd.py"]
