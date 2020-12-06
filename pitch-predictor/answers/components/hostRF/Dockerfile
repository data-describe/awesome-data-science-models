FROM google/cloud-sdk:latest



RUN pip3 installpandas
RUN pip3 installgoogle-cloud-storage


COPY ./ /root/

RUN chmod 777 -R /root
ENTRYPOINT [ "python", "/root/host_rf.py" ]