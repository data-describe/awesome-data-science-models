FROM google/cloud-sdk:latest



RUN pip3 install pandas
RUN pip3 install google-cloud-storage


COPY ./ /root/

RUN chmod 777 -R /root
ENTRYPOINT [ "python", "/root/host_rf.py" ]