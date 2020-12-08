FROM google/cloud-sdk:latest

RUN pip3 install --upgrade pip

RUN pip3 install pandas
RUN pip3 install google-cloud-storage
RUN pip3 install google-api-python-client

WORKDIR /root
COPY . .

ENTRYPOINT [ "python3", "enhance_features.py" ]