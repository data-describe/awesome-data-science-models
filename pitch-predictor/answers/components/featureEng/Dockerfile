FROM google/cloud-sdk:latest

ENV GCP_PROJECT mwpmltr

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY ./ /root/

RUN chmod 777 -R /root
ENTRYPOINT [ "python", "/root/feature_eng.py" ]