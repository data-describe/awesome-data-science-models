import googleapiclient.discovery
import pandas as pd

def returnPredEnhanced(pitch_data):
    service = googleapiclient.discovery.build('ml', 'v1')
    pitch_dict = {0:'FT',1:'FS',2:'CH',3:'FF',4:'SL',5:'CU',6:'FC',7:'SI',8:'KC',9:'EP',10:'KN',11:'FO'}
    

    MODEL_NAME = 'RFensemble' 
    name = 'projects/ross-kubeflow/models/{}'.format(MODEL_NAME)

    response = service.projects().predict(
            name=name,
            body={'instances': pitch_data}
        ).execute()

    pred = response['predictions'][0]

    pitch_pred = pitch_dict[pred]
        
    return pitch_pred