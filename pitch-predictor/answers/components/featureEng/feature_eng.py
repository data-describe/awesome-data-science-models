# libraries
from google.cloud import storage
import logging
import pandas_gbq


def run(argv=None):


    SQL = """
    SELECT  
     sz_top
    ,sz_bot
    ,pfx_xDataFile
    ,pfx_zDataFile
    ,zone_location
    ,pitch_con
    ,spin
    ,norm_ht
    ,tstart
    ,vystart
    ,ftime
    ,pfx_x
    ,pfx_z
    ,uncorrected_pfx_x
    ,uncorrected_pfx_z
    ,x0
    ,y0
    ,z0
    ,vx0
    ,vy0
    ,vz0
    ,ax
    ,ay
    ,az
    ,start_speed
    ,px
    ,pz
    ,pxold
    ,pzold
    ,tm_spin
    ,sb
    ,CASE WHEN mlbam_pitch_name = 'FT' THEN 1 ELSE 0 END AS FT
    ,CASE WHEN mlbam_pitch_name = 'FS' THEN 1 ELSE 0 END AS FS
    ,CASE WHEN mlbam_pitch_name = 'CH' THEN 1 ELSE 0 END AS CH
    ,CASE WHEN mlbam_pitch_name = 'FF' THEN 1 ELSE 0 END AS FF
    ,CASE WHEN mlbam_pitch_name = 'SL' THEN 1 ELSE 0 END AS SL
    ,CASE WHEN mlbam_pitch_name = 'CU' THEN 1 ELSE 0 END AS CU
    ,CASE WHEN mlbam_pitch_name = 'FC' THEN 1 ELSE 0 END AS FC
    ,CASE WHEN mlbam_pitch_name = 'SI' THEN 1 ELSE 0 END AS SI
    ,CASE WHEN mlbam_pitch_name = 'KC' THEN 1 ELSE 0 END AS KC
    ,CASE WHEN mlbam_pitch_name = 'EP' THEN 1 ELSE 0 END AS EP
    ,CASE WHEN mlbam_pitch_name = 'KN' THEN 1 ELSE 0 END AS KN
    ,CASE WHEN mlbam_pitch_name = 'FO' THEN 1 ELSE 0 END AS FO

    FROM `ross-kubeflow.baseball.raw_games`
    """

    df = pandas_gbq.read_gbq(SQL, project_id='ross-kubeflow')

    storage_client = storage.Client()
    bucket_name = 'raw-pitch-data'
    destination_blob_name = 'metrics.csv'

    source_file_name = 'metrics.csv'
    df.to_csv(source_file_name,index=False)

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    run()



