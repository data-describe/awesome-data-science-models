FROM google/cloud-sdk:latest



RUN pip3 installpandas
RUN pip3 installgoogle-cloud-storage
RUN pip3 installnumpy
RUN pip3 installjoblib
RUN pip3 installsklearn


COPY ./ /root/

RUN chmod 777 -R /root
ENTRYPOINT [ "python", "/root/train_rf.py" ]