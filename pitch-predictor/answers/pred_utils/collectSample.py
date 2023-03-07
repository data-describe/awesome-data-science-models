import pandas_gbq

def collectSample():
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

    FROM `{{ GCP_PROJECT }}.baseball.raw_games`
    WHERE mlbam_pitch_name IN ('FT','FS','CH','FF','SL','CU','FC','SI','KC','EP','KN','FO')
    LIMIT 100
    """
    
    sample_df = pandas_gbq.read_gbq(SQL, project_id='{{ GCP_PROJECT }}')
    
    return sample_df