from kfp.v2.dsl import component


@component(packages_to_install=["google-cloud-storage==1.44.0"])
def validate_file_type(bucket_name: str) -> list:
    """Pipeline component that validates the files"""

    from google.cloud import storage

    storage_client = storage.Client()

    def _get_blobs_in(bucket_name: str, prefix: str) -> list:
        """Gets a list of all the blobs in the bucket"""
        blobs = storage_client.list_blobs(
            bucket_name, prefix=prefix
        )  # returns only first level objects
        blob_names = [blob.name for blob in blobs if not blob.name.endswith("/")]
        return blob_names

    def _validate_file_type(
        blob_name: str, valid_file_endings: list = [".pdf"]
    ) -> bool:
        """Validate file types"""
        for file_ending in valid_file_endings:
            if blob_name.endswith(file_ending):
                return True

        # if not one of the valid file types
        return False

    def _copy_blob(bucket_name: str, blob_name: str, destination_prefix: str) -> None:
        """Copies blob to different folder in bucket"""
        source_bucket = storage_client.bucket(bucket_name)
        source_blob = source_bucket.blob(blob_name)
        destination_bucket = source_bucket  # just moving between folders, not buckets

        file_name = blob_name.split("/")[1]
        destination_blob_name = f"{destination_prefix}/{file_name}"

        copy_response = source_bucket.copy_blob(
            source_blob, destination_bucket, destination_blob_name
        )
        return None

    documents_in_lake = _get_blobs_in(bucket_name, prefix="document_lake")
    valid_documents = []
    invalid_documents = []
    for document in documents_in_lake:
        is_valid = _validate_file_type(document)
        if is_valid:
            valid_documents.append(document)
            _ = _copy_blob(bucket_name, document, "valid_file_type")
        else:
            invalid_documents.append(document)
            _ = _copy_blob(bucket_name, document, "dead_letters/invalid_file_type")

    return valid_documents
