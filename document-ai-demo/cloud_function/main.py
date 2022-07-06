import sys
from flask import escape
import functions_framework
import pandas as pd
from google.cloud import documentai_v1 as documentai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import pickle
from google.cloud import storage
from google.cloud import firestore
import json

# set parameters
project_id= 'mwpmltr'
location = 'us' # Format is 'us' or 'eu'
processor_id = '5146e2e343bf6d70' # document ocr #  Create processor in Cloud Console
mime_type = 'application/pdf'

def process_document(project_id: str, location: str,
                     processor_id: str, file_path: str,
                     mime_type: str) -> documentai.Document:
    """
    Processes a document using the Document AI API.
    """

    # Instantiates a client
    documentai_client = documentai.DocumentProcessorServiceClient()

    # The full resource name of the processor, e.g.:
    # projects/project-id/locations/location/processor/processor-id
    # You must create new processors in the Cloud Console first
    resource_name = documentai_client.processor_path(
        project_id, location, processor_id)

    # Read the file into memory
    with open(file_path, "rb") as image:
        image_content = image.read()

        # Load Binary Data into Document AI RawDocument Object
        raw_document = documentai.RawDocument(
            content=image_content, mime_type=mime_type)

        # Configure the process request
        request = documentai.ProcessRequest(
            name=resource_name, raw_document=raw_document)

        # Use the Document AI client to process the sample form
        result = documentai_client.process_document(request=request)

        return result.document
    
# perform Key Value pair
#Set up processor variables
PROJECT_ID = project_id
LOCATION = "us"  # Format is 'us' or 'eu'

#initialize firestore client
db = firestore.Client(project='mwpmltr')

def process_document_sample(PROCESSOR_ID,PDF_PATH):
    # Instantiates a client
    client_options = {"api_endpoint": "{}-documentai.googleapis.com".format(LOCATION)}
    client = documentai.DocumentProcessorServiceClient(client_options=client_options)

    # The full resource name of the processor, e.g.:
    # projects/project-id/locations/location/processor/processor-id
    # You must create new processors in the Cloud Console first
    name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"

    with open(PDF_PATH, "rb") as image:
        image_content = image.read()

    # Read the file into memory
    document = {"content": image_content, "mime_type": "application/pdf"}

    # Configure the process request
    request = {"name": name, "raw_document": document}

    # Recognizes text entities in the PDF document
    result = client.process_document(request=request)
    document = result.document
    entities = document.entities
    print("Document processing complete.\n\n")

    # For a full list of Document object attributes, please reference this page: https://googleapis.dev/python/documentai/latest/_modules/google/cloud/documentai_v1beta3/types/document.html#Document  
    types = []
    values = []
    confidence = []
    
    # Grab each key/value pair and their corresponding confidence scores.
    for entity in entities:
        types.append(entity.type_)
        values.append(entity.mention_text)
        confidence.append(round(entity.confidence,4))
        
    # Create a Pandas Dataframe to print the values in tabular format. 
    df = pd.DataFrame({'Type': types, 'Value': values, 'Confidence': confidence})
    
    return document,df


def get_text(doc_element: dict, document: dict):
    """
    Document AI identifies form fields by their offsets
    in document text. This function converts offsets
    to text snippets.
    """
    response = ""
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    for segment in doc_element.text_anchor.text_segments:
        start_index = (
            int(segment.start_index)
            if segment in doc_element.text_anchor.text_segments
            else 0
        )
        end_index = int(segment.end_index)
        response += document.text[start_index:end_index]
    return response    

def save_output_firestore(PROCESSOR_ID,document_file_path,blobName,prediction):
    doc,dataframe = process_document_sample(PROCESSOR_ID,document_file_path)

    # write data to firestore
    json_output = dataframe.to_json(orient='index')
    json_output=json.loads(json_output)
    json_output["document type"]=prediction
    print(json_output)
    doc_ref = db.collection(u'DocAI').document(u'document-'+str(blobName))
    doc_ref.set(json_output)


def Doc_AI(event, context):
    bucketName = event['bucket']
    blobName = event['name']
    fileName = "gs://" + bucketName + "/" + blobName
    print("filename is ", fileName)
    
    # download file
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucketName)
    
    blob = bucket.blob(blobName)
    contents = blob.download_as_string()
    
    document_file_path='/tmp/document_file.pdf'
    blob.download_to_filename(document_file_path)


    
    # extract text from the document
    
    document = process_document(project_id=project_id, location=location,
                            processor_id=processor_id, file_path=document_file_path,
                            mime_type=mime_type)

    
    text=[" ".join(document.text.splitlines()).strip()]
    print("the text is ", text)
    
    
    # read the pipeline from pickle
    # download file
    #bucketName = 'mwpmltr-1'
    #blobName_pipeline = 'classification_pipeline.pickle'
    
    bucketName = 'demo-document-intelligence'
    blobName_pipeline = 'models/classification_pipeline.pickle'

    bucket = storage_client.bucket(bucketName)
    
    blob = bucket.blob(blobName_pipeline)
    contents = blob.download_as_string()
    
    pipeline_file_path='/tmp/classification_pipeline.pickle'
    blob.download_to_filename(pipeline_file_path)

    pipeline=pickle.load(open(pipeline_file_path, 'rb'))
    prediction = str(list(pipeline.predict(text))[0])
    print(prediction)

    # key value pair extraction for w-9 and invoice | write to firestore
    if prediction=='w-9' or prediction=='w9':
        PROCESSOR_ID ='ece6aa307cb0d855'
        save_output_firestore(PROCESSOR_ID,document_file_path,blobName,prediction)
    elif prediction=='invoice':
        PROCESSOR_ID ='ecb83bc045cbbb9c'
        save_output_firestore(PROCESSOR_ID,document_file_path,blobName,prediction)
    else:
        dictionary ={"document type" : prediction}
        doc_ref = db.collection(u'DocAI').document(u'document-'+str(blobName))
        doc_ref.set(dictionary)
    
    