FROM google/cloud-sdk:latest
RUN pip install pandas
RUN pip install google-cloud-storage


COPY ./ /root/

RUN chmod 777 -R /root
ENTRYPOINT [ "python", "/root/host_xgboost.py" ]