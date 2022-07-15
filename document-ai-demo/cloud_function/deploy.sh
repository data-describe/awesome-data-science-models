gcloud functions deploy Doc_AI \
    --project mwpmltr \
    --entry-point process_document \
    --runtime python39 \
    --trigger-bucket docai_upload_document \
    --memory 8192MB \
    --max-instances 3000