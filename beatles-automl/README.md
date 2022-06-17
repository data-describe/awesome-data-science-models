# The Beatles Machine Learning Data set

Data for use on Machine Learning Model to predict those who will listen The Beatles based on other artists. There are two data sets of importance:

* [file_out_2495.csv](./answers/file_out_2495.csv) a list of users who listened to at least 1 of the most 300 played artists. The columns are the play counts for each artists mentioned. The target is "Likes the Beatles"
* [file_out_2495_tags.csv](./answers/file_out_2495_tags.csv) Same as above but with also a count of the genre distribution. 


Instructions to re-generate w/ tags:

1. Get a GCP Account and open the Jupyter notebook in Platform AI or DataLab
2. Get a lastfm API account and edit the [lastfm.conf](./answers/lastfm.conf)
3. Enable the free https://console.cloud.google.com/marketplace/details/metabrainz/listenbrainz database in BigQuery
4. Run the code to get the data from BigQuery [Data from BiqQuery.ipynb](./answers/Data%20from%20BiqQuery.ipynb)
4. Run the code in [Enrich_top_300.ipynb](./answers/Enrich_top_300.ipynb)
5. Run the code in [listen_top_300.ipynb](./answers/listen_top_300.ipynb)


The output file will be called [file_out_2495_tags.csv](./answers/file_out_2495_tags.csv)

Please use this link to follow the AutoMl tutorial 
https://towardsdatascience.com/build-a-useful-ml-model-in-hours-on-gcp-to-predict-the-beatles-listeners-1b2322486bdf

Questions: Brianhray@gmail.com


