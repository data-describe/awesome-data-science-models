FROM google/cloud-sdk:latest
RUN pip install pandas
RUN pip install numpy
RUN pip install google-cloud-storage

COPY ./ /root/

RUN chmod 777 -R /root
ENTRYPOINT [ "python", "/root/split_train_test_val.py" ]