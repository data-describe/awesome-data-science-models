FROM google/cloud-sdk:latest

RUN pip3 install --upgrade pip

RUN pip3 install pandas
RUN pip3 install google-cloud-storage


WORKDIR /root
COPY . .

ENTRYPOINT [ "python3", "host_rf.py" ]