from kfp.v2.dsl import component


@component(packages_to_install=["google-cloud-storage==1.44.0"])
def validate_class(bucket_name: str) -> None:
    """Pipeline component that extracts and validates the labelled document class"""

    from google.cloud import storage

    storage_client = storage.Client()

    def _get_blobs_in(bucket_name: str, prefix: str) -> list:
        """Gets a list of all the blobs in the bucket"""
        blobs = storage_client.list_blobs(
            bucket_name, prefix=prefix
        )  # returns only first level objects
        blob_names = [blob.name for blob in blobs if not blob.name.endswith("/")]
        return blob_names

    def _extract_class_name(
        blob_name: str, valid_classes: list = ["1040", "1099r", "1120", "w9"]
    ) -> str:
        """Infers the class of document from the file name"""

        class_name = (
            blob_name.replace("valid_file_type/", "").split("_")[0].replace("f", "")
        )

        for document_class in valid_classes:
            if class_name == document_class:
                return class_name

        # if not one of the valid classes
        return None

    def _copy_blob(bucket_name: str, blob_name: str, destination_prefix: str) -> None:
        """Copies blob to different folder in bucket"""
        source_bucket = storage_client.bucket(bucket_name)
        source_blob = source_bucket.blob(blob_name)
        destination_bucket = source_bucket  # just moving between folders, not buckets

        file_name = blob_name.split("/")[1]
        destination_blob_name = f"{destination_prefix}/{file_name}"

        _ = source_bucket.copy_blob(
            source_blob, destination_bucket, destination_blob_name
        )
        return None

    documents_to_organize = _get_blobs_in(bucket_name, "valid_file_type")
    documents_with_valid_class = []
    documents_with_unknown_class = []
    for document in documents_to_organize:
        class_name = _extract_class_name(document)

        if class_name is not None:
            documents_with_valid_class.append(
                {"blob_name": document, "label": class_name}
            )
            _ = _copy_blob(bucket_name, document, f"document_cohort/{class_name}")
        else:
            documents_with_unknown_class.append(document)
            _ = _copy_blob(bucket_name, document, "dead_letters/unknown_class")
