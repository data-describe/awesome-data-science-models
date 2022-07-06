# Kubeflow Pipelines
This subfolder houses the definition of the Kubeflow pipeline used for training the Random Forest document classification model.

## Setup
This pipeline defined in `kubeflow.py` is a [Kubeflow Pipeline](https://www.kubeflow.org/docs/components/pipelines/introduction/) that is configured to run on the Google Cloud Platform via [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction). In order to setup your environment to deploy this pipeline, you'll need to install the packages outlined in `requirements.txt`.

Assuming you have a Conda environment named `demo-document-ai`, installing this packages would look something like this:
```bash
conda activate demo-document-ai
pip install -r requirements.txt
```

### Authentication
You'll need to make sure that you have the [gcloud cli](https://cloud.google.com/sdk/gcloud) set up on your machine. One the **cli** is installed, you should run `gcloud auth application-default login` and follow the instructions to generate a set of default Google Cloud credentials. These credentials will be used every time you attempt to make API calls to Google Cloud resources.

Prior to generating those default credentials you should ensure that your Google Cloud identity has the proper access to create pipeline runs. See the [Vertex AI IAM permissions documentation](https://cloud.google.com/vertex-ai/docs/general/iam-permissions) for more details.

## Making Changes
In order to make changes to the Kubeflow components, you'll need to modify one or more of the corresponding files housed in the `components` folder. You can also make changes to the pipeline configuration by modifying the `kubeflow.py` file.

## Creating Pipeline Runs
In order to create a pipeline run, activate your virtual environment and run `python kubeflow.py`.

## Authors
Claire B. Salling - claire.salling@mavenwave.com
