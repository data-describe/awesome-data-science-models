from google.cloud import storage
import logging
import pandas as pd
import pandas_gbq

def to_target_feature(df, cat_1s):
    # collect category 1 vector
    cat1_dict = df['Product_Category_1'].value_counts().to_dict()
    
    user_cat1_vec = []
    for cat in cat_1s:
        try:
            val = cat1_dict[cat]
        except:
            val = 0
        user_cat1_vec.append(val)
    
    return user_cat1_vec

def to_categorical_features(df, cat_1s, cat_2s, cat_3s, occupations, city_cats):
    # collect category 1 vector
    cat1_dict = df['Product_Category_1'].value_counts().to_dict()
    
    user_cat1_vec = []
    for cat in cat_1s:
        try:
            val = cat1_dict[cat]
        except:
            val = 0
        user_cat1_vec.append(val)
    
    # collect category 2 vector
    cat2_dict = df['Product_Category_2'].value_counts().to_dict()

    user_cat2_vec = []
    for cat in cat_2s:
        try:
            val = cat2_dict[cat]
        except:
            val = 0
        user_cat2_vec.append(val)

    # collect category 3 vector
    cat3_dict = df['Product_Category_3'].value_counts().to_dict()

    user_cat3_vec = []
    for cat in cat_3s:
        try:
            val = cat3_dict[cat]
        except:
            val = 0
        user_cat3_vec.append(val)

    # collect occupation vector
    occupation = df['Occupation'][df.first_valid_index()]
    occupation_vec = []
    for occ in occupations:
        if occ == occupation:
            occupation_vec.append(1)
        else:
            occupation_vec.append(0)
    
    # collect city category vector
    city = df['City_Category'][df.first_valid_index()]
    city_vec = []
    for cat in city_cats:
        if cat == city:
            city_vec.append(1)
        else:
            city_vec.append(0)
    
    user_cat_vector = user_cat1_vec + user_cat2_vec + user_cat3_vec + occupation_vec + city_vec
                            
    return user_cat_vector

def produce_features(df, cat_1s, cat_2s, cat_3s, occupations, city_cats, datapart, bucket):
    x_rows = []
    y_rows = []
    
    for user_id in set(df['User_ID']):
        df_part = df[df['User_ID'] == user_id].reset_index(drop=True)
        
        df_part_y = pd.DataFrame(df_part.iloc[0]).T # just the 1st row
        df_part_x = df_part.drop([0]) # drop the 1st row
         
        
        # transform categorical features
        user_cat_vector = to_categorical_features(df_part_x, cat_1s, cat_2s, cat_3s, occupations, city_cats)
        
        # collect continuous features
        first_valid_index = df_part_x.first_valid_index()

        user_id = df_part_x['User_ID'][first_valid_index]
        gender = df_part_x['Gender'][first_valid_index]
        age = df_part_x['Age'][first_valid_index]
        stay = df_part_x['Stay_In_Current_City_Years'][first_valid_index]
        marital = df_part_x['Marital_Status'][first_valid_index]

        continuous_features = [gender, age, stay, marital]
        
        # combine the continuous and categorial
        x_row = continuous_features + user_cat_vector
        
        # collect the target vector
        target = to_target_feature(df_part_y, cat_1s)
        
        # combine with the user_id
        y_row = target
        
        # add results to the list
        x_rows.append(x_row)
        y_rows.append(y_row)


    # save results to Cloud Storage
        # x
    df_x = pd.DataFrame(x_rows)
    filename = 'x_{}.csv'.format(datapart)
    df_x.to_csv(filename, index=False, header=False)
    blob = bucket.blob(filename)
    blob.upload_from_filename(filename)

        # y
    df_y = pd.DataFrame(y_rows)
    filename = 'y_{}.csv'.format(datapart)
    df_y.to_csv(filename, index=False, header=False)
    blob = bucket.blob(filename)
    blob.upload_from_filename(filename)


def create_data_func(project_id, bucket_name, dataset_id):
    # download the train and test files
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.get_bucket(bucket_name) 

    blob = bucket.blob('train.csv')
    blob.download_to_filename('train.csv')
    train_df = pd.read_csv('train.csv')

    blob = bucket.blob('test.csv')
    blob.download_to_filename('test.csv')
    test_df = pd.read_csv('test.csv')

    # remove Purchase column from test set and combine
    train_df = train_df.drop(['Purchase'],axis=1)
    full_df = pd.concat([train_df,test_df])

    # load the combined data to BigQuery
    full_df.to_gbq(destination_table='{}.full_dataset_raw'.format(dataset_id), project_id=project_id, if_exists='replace')


    # perform feature transformation and train test split
    sql = """
    SELECT 
     User_ID
    ,CASE WHEN Gender = 'M' THEN 1 ELSE 0 END AS Gender
    ,CASE
      WHEN Age = '0-17' THEN 1
      WHEN Age = '18-25' THEN 2
      WHEN Age = '26-35' THEN 3
      WHEN Age = '36-45' THEN 4
      WHEN Age = '46-50' THEN 5
      WHEN Age = '51-55' THEN 6
      WHEN Age = '55+' THEN 7
      ELSE 0
      END AS Age
    ,Occupation
    ,City_Category
    ,CASE WHEN Stay_In_Current_City_Years = '4+' THEN 4 ELSE CAST(Stay_In_Current_City_Years AS INT64) END AS Stay_In_Current_City_Years
    ,Marital_Status
    ,IFNULL(Product_Category_1,0) AS Product_Category_1
    ,IFNULL(Product_Category_2,0) AS Product_Category_2
    ,IFNULL(Product_Category_3,0) AS Product_Category_3
    FROM `{}.{}.full_dataset_raw` 
    """

    train_sql = sql.format(project_id, dataset_id) + "WHERE MOD(User_ID, 5) != 0 -- 80% to train "
    test_sql = sql.format(project_id, dataset_id) + "WHERE MOD(User_ID, 5) = 0 -- 20% to test"

    # Limit data size for demo purposes.
    train_sql += "\nLIMIT 1000"
    test_sql += "\nLIMIT 1000"    
    
    train_df = pandas_gbq.read_gbq(train_sql, project_id=project_id)
    test_df = pandas_gbq.read_gbq(test_sql, project_id=project_id)

    # collect all possible categorical values
    cat_1s = sorted(set(train_df['Product_Category_1']))
    cat_2s = sorted(set(train_df['Product_Category_2']))
    cat_3s = sorted(set(train_df['Product_Category_3']))
    occupations = sorted(set(train_df['Occupation']))
    city_cats = sorted(set(train_df['City_Category']))

    # generate the feature sets
    logging.info('Generating train set.')
    produce_features(df=train_df, cat_1s=cat_1s, cat_2s=cat_2s, cat_3s=cat_3s, occupations=occupations, city_cats=city_cats, datapart='train', bucket=bucket)
    logging.info('Generating test set.')
    produce_features(df=test_df, cat_1s=cat_1s, cat_2s=cat_2s, cat_3s=cat_3s, occupations=occupations, city_cats=city_cats, datapart='test', bucket=bucket)

    

