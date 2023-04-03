FROM gcr.io/deeplearning-platform-release/base-cpu

WORKDIR /

# Installs hypertune library
RUN pip install pandas numpy cloudml-hypertune scikit-learn xgboost

# Copies the trainer code to the Docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]