FROM google/cloud-sdk:latest
RUN pip install pandas_gbq
RUN pip install google-cloud-storage

COPY ./ /root/

RUN chmod 777 -R /root
ENTRYPOINT [ "python", "/root/feature_eng.py" ]