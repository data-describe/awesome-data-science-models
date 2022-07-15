from kfp.v2.dsl import component


@component(packages_to_install=["google-cloud-storage==1.44.0", "scikit-learn==1.0.2"])
def fit_classifier(project_id: str, bucket_name: str) -> None:
    """Kubeflow component to fit a Random Forest classification model"""

    from google.cloud import storage
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.pipeline import make_pipeline
    import pickle
    import logging

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

    def _infer_classes_of(training_data: list) -> list:
        """Infers the class names from the training data file names"""
        labelled_classes = []
        for data_point in training_data:
            class_name = data_point.split("/")[1]
            labelled_classes.append(class_name)

        return labelled_classes

    def _fit_model(training_data: list, training_labels: list) -> Pipeline:
        """Fits a Random Forest classifier to the input text data and classes"""
        vectorizer = TfidfVectorizer()
        model = RandomForestClassifier(n_estimators=100)
        pipeline = make_pipeline(vectorizer, model)
        pipeline.fit(training_data, training_labels)

        return pipeline

    extracted_text_blobs = _get_blobs_in(bucket_name, prefix="extracted_text")
    logging.info("Blobs with OCR Results: {}".format(extracted_text_blobs[0]))

    labelled_classes = _infer_classes_of(extracted_text_blobs)
    logging.info("Inferred classes include: {}".format(set(labelled_classes)))

    all_extracted_text = []
    for text_file in extracted_text_blobs:
        _ = _download_blob(bucket_name, text_file, "temp_download.txt")
        with open("temp_download.txt", "r") as infile:
            extracted_text = infile.read()
            all_extracted_text.append(extracted_text)

    pipeline = _fit_model(all_extracted_text, labelled_classes)

    _ = pickle.dump(pipeline, open("classification_pipeline.pickle", "wb"))
    _ = _upload_blob(
        bucket_name,
        "classification_pipeline.pickle",
        "models/classification_pipeline.pickle",
    )
