# Kubeflow Pipelines Demo Using Autoencoder for Anomaly Detection
This repo explores the use case of applying IOT sensor data to train and an unsupervised, autoencoder model for the use of anomaly detection within a Kubeflow pipeline.

The model used in this demonstration is an LSTM model using the NASA Bearing Sensor dataset based off of the following article and repo:

1. [Medium Article Discussing LSTM-based Autoencoders](https://towardsdatascience.com/lstm-autoencoder-for-anomaly-detection-e1f4f2ee7ccf)
2. [Git Repo for the Article](https://github.com/BLarzalere/LSTM-Autoencoder-for-Anomaly-Detection/blob/master/Sensor%20Anomaly%20Detection.ipynb)

### Kubeflow Additional Resources:
1. [DSL Control Structures (conditional logic)](https://github.com/kubeflow/pipelines/tree/b604c6171244cc1cd80bfdc46248eaebf5f985d6/samples/tutorials/DSL%20-%20Control%20structures)
2. [Data Passing Components](https://github.com/kubeflow/pipelines/tree/b604c6171244cc1cd80bfdc46248eaebf5f985d6/samples/tutorials/Data%20passing%20in%20python%20components)
3. [Kubeflow Pipeline Samples](https://github.com/kubeflow/pipelines/tree/b604c6171244cc1cd80bfdc46248eaebf5f985d6/samples)
4. [Kubeflow Pipelines SDK Intro](https://www.kubeflow.org/docs/pipelines/sdk/sdk-overview/)
5. [Push/Pull Containers on GCR](https://cloud.google.com/container-registry/docs/pushing-and-pulling)