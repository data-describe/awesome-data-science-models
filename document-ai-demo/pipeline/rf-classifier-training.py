import kfp.dsl as dsl
from kfp.v2 import compiler
from google.cloud import aiplatform

from components.validate_file_type import validate_file_type
from components.validate_class import validate_class
from components.extract_text import extract_text
from components.fit_classifier import fit_classifier

ENV = "dev"
PROJECT_ID = "mwpmltr"
PIPELINE_ROOT_PATH = "gs://demo-document-intelligence/kubeflow"
BUCKET_NAME = "demo-document-intelligence"
PROCESSOR_ID = "5146e2e343bf6d70"


@dsl.pipeline(
    name=f"{ENV}-document-classification-pipeline",
    description="A training pipeline for creating a document classification model",
    pipeline_root=PIPELINE_ROOT_PATH,
)
def add_pipeline(project_id: str, bucket_name: str, processor_id: str):
    _validate_file_type_op = validate_file_type(bucket_name).set_display_name(
        "Validate File Type"
    )
    _validate_class_op = (
        validate_class(bucket_name)
        .after(_validate_file_type_op)
        .set_display_name("Validate Document Class")
    )
    _extract_text_op = (
        extract_text(project_id, bucket_name, processor_id)
        .after(_validate_class_op)
        .set_display_name("OCR Text Extraction")
    )
    _fit_classifier_op = (
        fit_classifier(project_id, bucket_name)
        .after(_extract_text_op)
        .set_display_name("Fit RF Classifier")
    )


if __name__ == "__main__":

    compiler.Compiler().compile(
        pipeline_func=add_pipeline, package_path=f"{ENV}_docai_pipeline.json"
    )

    job = aiplatform.PipelineJob(
        display_name=f"{ENV}_docai_pipeline",
        template_path=f"{ENV}_docai_pipeline.json",
        pipeline_root=PIPELINE_ROOT_PATH,
        parameter_values={
            "project_id": PROJECT_ID,
            "bucket_name": BUCKET_NAME,
            "processor_id": PROCESSOR_ID,
        },
    )

    job.submit()
