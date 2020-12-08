import os
from func_components import load_raw_data
from func_components import split_data
from func_components import disp_loss
from jinja2 import Template
import kfp
from kfp.components import func_to_container_op
from kfp.dsl.types import Dict
from kfp.dsl.types import GCPProjectID
from kfp.dsl.types import GCPRegion
from kfp.dsl.types import GCSPath
from kfp.dsl.types import String
from kfp.gcp import use_gcp_secret

# Defaults and environment settings
BASE_IMAGE = os.getenv('BASE_IMAGE')
TRAINER_IMAGE = os.getenv('TRAINER_IMAGE')
DD_IMAGE = os.getenv('DD_IMAGE')
RUNTIME_VERSION = os.getenv('RUNTIME_VERSION')
PYTHON_VERSION = os.getenv('PYTHON_VERSION')
COMPONENT_URL_SEARCH_PREFIX = os.getenv('COMPONENT_URL_SEARCH_PREFIX')
USE_KFP_SA = os.getenv('USE_KFP_SA')

# Create component factories
component_store = kfp.components.ComponentStore(
    local_search_paths=None, url_search_prefixes=[COMPONENT_URL_SEARCH_PREFIX])

# Create all the component ops
caip_train_op = component_store.load_component('ml_engine/train')

retrieve_raw_data_op = func_to_container_op(
    load_raw_data, base_image=BASE_IMAGE)

split_preprocess_data_op = func_to_container_op(
    split_data, base_image=BASE_IMAGE)

disp_loss_op = func_to_container_op(
    disp_loss)

def datadescribe_op(gcs_root, filepath):
    return kfp.dsl.ContainerOp(
        name='Run_Data_Decsribe',
        image = 'gcr.io/mwpmltr/rrusson_kubeflow_datadescribe:v1',
        arguments=[
            '--gcs_root', gcs_root,
            '--file', filepath
        ]
    )


# Define the pipeline
@kfp.dsl.pipeline(
    name='Bearing Sensor Data Training',
    description='The pipeline for training and deploying an anomaly detector based on an autoencoder')

def pipeline_run(project_id,
                 region,
                 source_bucket_name, 
                 prefix,
                 dest_bucket_name,
                 dest_file_name,
                 gcs_root="gs://rrusson-kubeflow-test",
                 dataset_location='US'):
    
    # Read in the raw sensor data from the public dataset and load in the project bucket
    raw_data = retrieve_raw_data_op(source_bucket_name,
                                    prefix,
                                    dest_bucket_name,
                                    dest_file_name)
    
    
    # Prepare some output from Data Describe
    dd_out = datadescribe_op(gcs_root, 
                             raw_data.outputs['dest_file_name'])
    
    
    # Preprocess and split the raw data by time
    split_data = split_preprocess_data_op(raw_data.outputs['dest_bucket_name'],
                                          raw_data.outputs['dest_file_name'],
                                          '2004-02-15 12:52:39',
                                          True)
    
    # Set up the training args
    train_args = ["--bucket", split_data.outputs['bucket_name'],
                  "--train_file", split_data.outputs['train_dest_file'],
                  "--test_file", split_data.outputs['test_dest_file']
                 ]
    
    job_dir = "{0}/{1}/{2}".format(gcs_root, 'jobdir', kfp.dsl.RUN_ID_PLACEHOLDER)
    
    # Train the model on AI Platform
    train_model = caip_train_op(project_id,
                                region=region,
                                master_image_uri=TRAINER_IMAGE,
                                job_id_prefix='anomaly-detection_',
                                job_dir=job_dir,
                                args=train_args)
    
    # Expose artifacts to the Kubeflow UI
    disp_loss_img = disp_loss_op(train_model.outputs['job_id'])
    disp_loss_dist_img = disp_loss_op(train_model.outputs['job_id'])
    
