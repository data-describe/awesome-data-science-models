FROM google/cloud-sdk:latest



RUN pip install pandas
RUN pip install google-cloud-storage
RUN pip install numpy
RUN pip install joblib
RUN pip install sklearn


COPY ./ /root/

RUN chmod 777 -R /root
ENTRYPOINT [ "python", "/root/train_rf.py" ]