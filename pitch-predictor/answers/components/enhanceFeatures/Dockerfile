FROM google/cloud-sdk:latest



RUN pip3 install pandas
RUN pip3 install google-cloud-storage
RUN pip3 install google-api-python-client

COPY ./ /root/

RUN chmod 777 -R /root
ENTRYPOINT [ "python", "/root/enhance_features.py" ]