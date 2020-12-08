import googleapiclient.discovery
import pandas as pd

def returnPred(pitch_data):
    service = googleapiclient.discovery.build('ml', 'v1')
    pitch_types = ['FT','FS','CH','FF','SL','CU','FC','SI','KC','EP','KN','FO']
    
    preds_dict = {}
    for pitch_type in pitch_types:
        MODEL_NAME = 'xgboost_' + pitch_type
        name = 'projects/{{ GCP_PROJECT }}/models/{}'.format(MODEL_NAME)

        response = service.projects().predict(
                name=name,
                body={'instances': pitch_data}
            ).execute()

        pred = response['predictions']

        preds_dict[pitch_type] = pred[0]
        
    pred_df = pd.DataFrame.from_dict(preds_dict,orient='index').transpose()
        
    return pred_df.style.highlight_max(axis=1)