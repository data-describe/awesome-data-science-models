FROM google/cloud-sdk:latest



RUN pip3 install pandas
RUN pip3 install google-cloud-storage
RUN pip3 install numpy
RUN pip3 install joblib
RUN pip3 install sklearn


COPY ./ /root/

RUN chmod 777 -R /root
ENTRYPOINT [ "python", "/root/train_rf.py" ]