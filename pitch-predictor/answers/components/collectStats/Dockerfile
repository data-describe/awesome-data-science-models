FROM apache/beam_python3.7_sdk

ENV GCP_PROJECT mwpmltr

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY ./ /root/


RUN chmod 777 -R /root
ENTRYPOINT [ "python3", "/root/collect_stats_dataflow.py" ]