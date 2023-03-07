import pandas as pd

def collectPitch(df):
    pitch_types = ['FT','FS','CH','FF','SL','CU','FC','SI','KC','EP','KN','FO']
    
    pitch = df.sample(n=1)
    
    pitch_label_df = pitch[pitch_types]
    
    pitch_data = pitch.drop(pitch_types,axis=1).values.tolist()
    
    return pitch_data, pitch_label_df.style.highlight_max(axis=1)