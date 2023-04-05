# The Beatles Machine Learning Data set

Data for use on Machine Learning Model to predict those who will listen The Beatles based on other artists. There are two data sets of importance:

* [file_out_2495.csv](./answers/file_out_2495.csv) a list of users who listened to at least 1 of the most 300 played artists. The columns are the play counts for each artists mentioned. The target is "Likes the Beatles"
* [file_out_2495_tags.csv](./answers/file_out_2495_tags.csv) Same as above but with also a count of the genre distribution. 


Instructions to re-generate w/ tags:

1. Get a GCP Account and open the Jupyter notebook in Platform AI or DataLab
2. Get a lastfm API account and edit the [lastfm.conf](./answers/lastfm.conf)
3. Enable the free https://console.cloud.google.com/marketplace/details/metabrainz/listenbrainz database in BigQuery
4. Run the code to get the data from BigQuery [0-Data_Ingestion_from_BigQuery.ipynb](./answers/0-Data_Ingestion_from_BigQuery.ipynb)
4. Run the code in [1-Enrich_Raw_Input_Data.ipynb](./answers/1-Enrich_Raw_Input_Data.ipynb)
5. Run the code in [2-Engineer_Training_Data.ipynb](./answers/2-Engineer_Training_Data.ipynb)
6. Run the code in [3-Train_AutoML_Model.ipynb](./answers/3-Train_AutoML_Model.ipynb)
7. Run the code in [4-Run_Batch_Inference.ipynb](./answers/4-Run_Batch_Inference.ipynb)
8. Run the code in [5-Deploy_and_Run_Online_Inference.ipynb](./answers/5-Deploy_and_Run_Online_Inference.ipynb)


The output file will be called [file_out_2495_tags.csv](./answers/file_out_2495_tags.csv)

Please use this link to follow the AutoMl tutorial 
https://towardsdatascience.com/build-a-useful-ml-model-in-hours-on-gcp-to-predict-the-beatles-listeners-1b2322486bdf

Questions: Brianhray@gmail.com


