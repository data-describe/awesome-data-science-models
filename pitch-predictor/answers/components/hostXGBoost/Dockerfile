FROM google/cloud-sdk:latest

RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /root

COPY . .

ENTRYPOINT [ "python3", "host_xgboost.py" ]