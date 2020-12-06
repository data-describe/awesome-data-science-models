FROM apache/beam_python3.7_sdk

COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /root
COPY . .

ENTRYPOINT [ "python3", "collect_stats_dataflow.py" ]