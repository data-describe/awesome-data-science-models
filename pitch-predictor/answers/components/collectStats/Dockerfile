FROM google/cloud-sdk:latest
RUN pip install apache_beam[gcp]

COPY ./ /root/

RUN chmod 777 -R /root
ENTRYPOINT [ "python", "/root/collect_stats_dataflow.py" ]