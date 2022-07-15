# Google Cloud Function
The Google Cloud Function Doc_AI is responsible for running inference (both classification and named entity recognition) on the documents uploaded to the `docai_upload_document` bucket.

## Getting Started
In order to get started working with the Google Cloud Function in this directory, you will first need to satisfy the following requirements:
- A virtual Python environment must be set up that runs Python 3.9. Guidance on working with virtual environments in Python can be found here: [Hitchhiker's Guide to Python](https://docs.python-guide.org/dev/virtualenvs/)
- You'll needed to install the required Python packages into that environment. Typically you would do this by activating the environment and then running `pip install -r requirements.txt`, however, it is recommended that you use the approach appropriate to the virtual environment manager of your choice.
- The Google Cloud CLI must also be installed on your system and configured to use `mwpmltr` as the default project. Instructions for installing the GCloud CLI can be found [here](https://cloud.google.com/sdk/docs/install).

## Deploying the Cloud Function
An new version of the `Doc_AI` Cloud Function can be deployed to Google Cloud by using the bash script stored in this directory. 

**CAUTION**: As of writing this, running the `deploy.sh` script in this way will overwrite the `Doc_AI` function currently in production. You may want to change the name referred to in the script to avoid overwriting the extant function and deploy a dev version instead. If you do accidentally overwrite the existing function, you can retrieve previous versions via Bitbucket source control or the method outlined in the Stack Overflow post [here](https://stackoverflow.com/questions/46797662/retrieving-an-old-version-of-a-google-cloud-function-source).

```bash
chmod +x deploy.sh
./deploy.sh
```

## Authors
* **Rohit Gupta** - rohit.gupta@mavenwave.com
* **Claire B. Salling** - claire.salling@mavenwave.com