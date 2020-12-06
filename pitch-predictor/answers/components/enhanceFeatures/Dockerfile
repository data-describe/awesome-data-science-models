FROM google/cloud-sdk:latest



RUN pip3 installpandas
RUN pip3 installgoogle-cloud-storage
RUN pip3 installgoogle-api-python-client

COPY ./ /root/

RUN chmod 777 -R /root
ENTRYPOINT [ "python", "/root/enhance_features.py" ]