FROM google/cloud-sdk:latest



COPY requirements.txt .
RUN pip3 install -r requirements.txt


COPY ./ /root/

RUN chmod 777 -R /root
ENTRYPOINT [ "python", "/root/find_threshold.py" ]