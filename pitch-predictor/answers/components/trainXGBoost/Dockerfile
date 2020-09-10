FROM google/cloud-sdk:latest
RUN pip install pandas
RUN pip install google-cloud-storage
RUN pip install xgboost


COPY ./ /root/

RUN chmod 777 -R /root
ENTRYPOINT [ "python", "/root/train_xgboost.py" ]