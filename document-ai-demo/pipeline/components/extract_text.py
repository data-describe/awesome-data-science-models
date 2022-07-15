from kfp.v2.dsl import component


@component(
    packages_to_install=[
        "google-cloud-storage==1.44.0",
        "google-cloud-documentai==1.4.1",
    ]
)
def extract_text(project_id: str, bucket_name: str, processor_id: str) -> list:
    """Pipeline component that sends documents to API for OCR"""

    from google.cloud import storage
    from google.cloud import documentai_v1 as documentai

    storage_client = storage.Client()

    def _get_blobs_in(bucket_name: str, prefix: str) -> list:
        """Gets a list of all the blobs in the bucket"""
        blobs = storage_client.list_blobs(
            bucket_name, prefix=prefix
        )  # returns only first level objects
        blob_names = [blob.name for blob in blobs if not blob.name.endswith("/")]
        return blob_names

    def _download_blob(
        bucket_name: str, source_blob_name: str, destination_file_name: str
    ) -> None:
        """Downloads a blob from the bucket."""
        bucket = storage_client.bucket(bucket_name)

        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

        return None

    def _upload_blob(
        bucket_name: str, source_file_name: str, destination_blob_name: str
    ) -> None:
        """Uploads a local file to Google Cloud Storage"""
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        return None

    def _process_document(
        bucket_name: str, blob_name: str, project_id: str, location: str, mime_type: str
    ) -> str:
        """
        Processes a document using the Document AI API.
        """
        documentai_client = documentai.DocumentProcessorServiceClient()

        resource_name = documentai_client.processor_path(
            project=project_id,
            location=location,
            processor=processor_id,
        )

        file_type = mime_type.split("/")[-1]
        local_file_name = f"temp_download.{file_type}"
        _ = _download_blob(bucket_name, blob_name, local_file_name)

        with open(local_file_name, "rb") as image:
            image_content = image.read()

        raw_document = documentai.RawDocument(
            content=image_content, mime_type=mime_type
        )
        request = documentai.ProcessRequest(
            name=resource_name, raw_document=raw_document
        )
        response = documentai_client.process_document(request=request)

        extracted_text = " ".join(response.document.text.splitlines()).strip()
        with open("temp_response.txt", "w") as outfile:
            outfile.write(extracted_text)

        destination_blob_name = blob_name.replace(
            "document_cohort", "extracted_text"
        ).replace(file_type, "txt")
        _ = _upload_blob(bucket_name, "temp_response.txt", destination_blob_name)

        return destination_blob_name

    documents_to_parse = _get_blobs_in(bucket_name, "document_cohort")
    ocr_results = []
    for document in documents_to_parse:
        file_type = document.split(".")[-1]

        if file_type == "pdf":
            mime_type = "application/pdf"
        elif file_type == "gif":
            mime_type = "image/gif"
        elif file_type == "tiff" or file_type == "tif":
            mime_type = "image/tiff"
        elif file_type == "jpg" or file_type == "jpeg":
            mime_type = "image/jpeg"
        elif file_type == "png":
            mime_type = "image/png"
        elif file_type == "bmp":
            mime_type = "image/bmp"
        elif file_type == "webp":
            mime_type = "image/webp"

        extracted_text = _process_document(
            bucket_name,
            blob_name=document,
            project_id=project_id,
            location="us",
            mime_type=mime_type,
        )
        ocr_results.append(extracted_text)

    return ocr_results
