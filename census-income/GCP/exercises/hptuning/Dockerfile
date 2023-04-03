FROM <choose-your-base-image>

WORKDIR /

# Install required dependencies e.g.
RUN pip install cloudml-hypertune

# Copies the trainer code to the Docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]