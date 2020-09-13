FROM google/cloud-sdk:latest
RUN pip install pandas
RUN pip install numpy
RUN pip install google-cloud-storage
RUN pip install hyperopt
RUN pip install xgboost
RUN pip install sklearn

COPY ./ /root/

RUN chmod 777 -R /root
ENTRYPOINT [ "python", "/root/tune_hp.py" ]