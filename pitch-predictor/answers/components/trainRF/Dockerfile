FROM google/cloud-sdk:latest

RUN pip3 install --upgrade pip

RUN pip3 install pandas
RUN pip3 install google-cloud-storage
RUN pip3 install numpy
RUN pip3 install joblib
RUN pip3 install sklearn

WORKDIR /root
COPY . .

ENTRYPOINT [ "python3", "train_rf.py" ]