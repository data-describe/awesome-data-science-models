from typing import Dict, List, Union, Optional

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from kfp.v2.dsl import (
    Artifact,
    Dataset,
    Input,
    InputPath,
    Model,
    Output,
    OutputPath,
    component,
)


@component(
    base_image="python:3.9",  # Use a different base image.
    packages_to_install=["pandas-gbq", "google-cloud-storage", "scikit-learn"],
)
def create_data(
    project_id: str,
    bucket_name: str,
    dataset_id: str,
    use_demographic: bool,
    train_file_x: OutputPath("csv"),
    train_file_y: OutputPath("csv"),
    test_file_x: OutputPath("csv"),
    test_file_y: OutputPath("csv"),
):

    from google.cloud import storage
    import logging
    import pandas as pd
    import pandas_gbq

    def to_target_feature(df, cat_1s):
        # collect category 1 vector
        cat1_dict = df["Product_Category_1"].value_counts().to_dict()

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
        cat1_dict = df["Product_Category_1"].value_counts().to_dict()

        user_cat1_vec = []
        for cat in cat_1s:
            try:
                val = cat1_dict[cat]
            except:
                val = 0
            user_cat1_vec.append(val)

        # collect category 2 vector
        cat2_dict = df["Product_Category_2"].value_counts().to_dict()

        user_cat2_vec = []
        for cat in cat_2s:
            try:
                val = cat2_dict[cat]
            except:
                val = 0
            user_cat2_vec.append(val)

        # collect category 3 vector
        cat3_dict = df["Product_Category_3"].value_counts().to_dict()

        user_cat3_vec = []
        for cat in cat_3s:
            try:
                val = cat3_dict[cat]
            except:
                val = 0
            user_cat3_vec.append(val)

        # collect occupation vector
        occupation = df["Occupation"][df.first_valid_index()]
        occupation_vec = []
        for occ in occupations:
            if occ == occupation:
                occupation_vec.append(1)
            else:
                occupation_vec.append(0)

        # collect city category vector
        city = df["City_Category"][df.first_valid_index()]
        city_vec = []
        for cat in city_cats:
            if cat == city:
                city_vec.append(1)
            else:
                city_vec.append(0)

        user_cat_vector = (
            user_cat1_vec + user_cat2_vec + user_cat3_vec + occupation_vec + city_vec
        )

        return user_cat_vector

    def produce_features(
        df,
        cat_1s,
        cat_2s,
        cat_3s,
        occupations,
        city_cats,
        datapart,
        bucket,
        use_demographic,
    ):
        x_rows = []
        y_rows = []

        for user_id in set(df["User_ID"]):
            df_part = df[df["User_ID"] == user_id].reset_index(drop=True)

            df_part_y = pd.DataFrame(df_part.iloc[0]).T  # just the 1st row
            df_part_x = df_part.drop([0])  # drop the 1st row

            # transform categorical features
            user_cat_vector = to_categorical_features(
                df_part_x, cat_1s, cat_2s, cat_3s, occupations, city_cats
            )

            # collect continuous features
            first_valid_index = df_part_x.first_valid_index()

            user_id = df_part_x["User_ID"][first_valid_index]
            gender = df_part_x["Gender"][first_valid_index]
            age = df_part_x["Age"][first_valid_index]
            stay = df_part_x["Stay_In_Current_City_Years"][first_valid_index]
            marital = df_part_x["Marital_Status"][first_valid_index]

            continuous_features = []
            if use_demographic is True:
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

        df_x = pd.DataFrame(x_rows)
        df_y = pd.DataFrame(y_rows)

        if datapart == "train":
            # filename = 'x_{}.csv'.format(datapart)
            # df_x.to_csv(filename, index=False, header=False)
            df_x.to_csv(train_file_x, index=False)
            blob = bucket.blob(train_file_x)
            blob.upload_from_filename(train_file_x)

            # filename = 'y_{}.csv'.format(datapart)
            df_y.to_csv(train_file_y, index=False)
            blob = bucket.blob(train_file_y)
            blob.upload_from_filename(train_file_y)
        else:
            df_x.to_csv(test_file_x, index=False)
            blob = bucket.blob(test_file_x)
            blob.upload_from_filename(test_file_x)

            # filename = 'y_{}.csv'.format(datapart)
            df_y.to_csv(test_file_y, index=False)
            blob = bucket.blob(test_file_y)
            blob.upload_from_filename(test_file_y)

    # download the train and test files
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob("train.csv")
    blob.download_to_filename("train.csv")
    train_df = pd.read_csv("train.csv")

    blob = bucket.blob("test.csv")
    blob.download_to_filename("test.csv")
    test_df = pd.read_csv("test.csv")

    # remove Purchase column from test set and combine
    train_df = train_df.drop(["Purchase"], axis=1)
    full_df = pd.concat([train_df, test_df])

    # load the combined data to BigQuery
    full_df.to_gbq(
        destination_table="{}.full_dataset_raw".format(dataset_id),
        project_id=project_id,
        if_exists="replace",
    )

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

    train_sql = (
        sql.format(project_id, dataset_id)
        + "WHERE MOD(User_ID, 5) != 0 -- 80% to train "
    )
    test_sql = (
        sql.format(project_id, dataset_id) + "WHERE MOD(User_ID, 5) = 0 -- 20% to test"
    )

    # Limit data size for demo purposes.
    # train_sql += "\nLIMIT 1001"
    # test_sql += "\nLIMIT 1000"

    train_df = pandas_gbq.read_gbq(train_sql, project_id=project_id)
    test_df = pandas_gbq.read_gbq(test_sql, project_id=project_id)

    # collect all possible categorical values
    cat_1s = sorted(set(train_df["Product_Category_1"]))
    cat_2s = sorted(set(train_df["Product_Category_2"]))
    cat_3s = sorted(set(train_df["Product_Category_3"]))
    occupations = sorted(set(train_df["Occupation"]))
    city_cats = sorted(set(train_df["City_Category"]))

    # generate the feature sets
    logging.info("Generating train set.")
    produce_features(
        df=train_df,
        cat_1s=cat_1s,
        cat_2s=cat_2s,
        cat_3s=cat_3s,
        occupations=occupations,
        city_cats=city_cats,
        datapart="train",
        bucket=bucket,
        use_demographic=use_demographic,
    )
    logging.info("Generating test set.")
    produce_features(
        df=test_df,
        cat_1s=cat_1s,
        cat_2s=cat_2s,
        cat_3s=cat_3s,
        occupations=occupations,
        city_cats=city_cats,
        datapart="test",
        bucket=bucket,
        use_demographic=use_demographic,
    )


@component(
    base_image="python:3.9",  # Use a different base image.
    packages_to_install=["google-cloud-storage", "scikit-learn", "pandas"],
)
def train_model(
    hp_tune: bool,
    project_id: str,
    bucket_name: str,
    train_file_x: InputPath("csv"),
    test_file_x: InputPath("csv"),
    train_file_y: InputPath("csv"),
    test_file_y: InputPath("csv"),
    best_params_file: OutputPath("json"),
    metrics_file: OutputPath("json"),
    model_file: Output[Model],
    num_iterations: Optional[int] = 2,
    best_params: Optional[dict] = None,
) -> None:

    from google.cloud import storage
    import json
    import logging
    import numpy as np
    import pandas as pd
    import os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import pickle
    import sklearn

    logging.info("The scikit-learn version is {}.".format(sklearn.__version__))

    metrics = {}

    x_train_df = pd.read_csv(train_file_x)
    x_test_df = pd.read_csv(test_file_x)
    y_test_df = pd.read_csv(test_file_y)
    y_train_df = pd.read_csv(train_file_y)

    if hp_tune is True:
        logging.info("Started hyperparameter tuning")
        best_accuracy = -1
        for i in range(0, num_iterations):
            # ramdom split for train and validation
            x_train, x_test, y_train, y_test = train_test_split(
                x_train_df, y_train_df, test_size=0.2
            )

            # randomly assign hyperparameters
            n_estimators = np.random.randint(10, 1000)
            max_depth = np.random.randint(10, 1000)
            min_samples_split = np.random.randint(2, 10)
            min_samples_leaf = np.random.randint(1, 10)
            max_features = ["auto", "sqrt", "log2", None][np.random.randint(0, 3)]

            # fit the model on the training set with the parameters
            rf_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=15,
                n_jobs=-1,
            )
            rf_model.fit(x_train, y_train)

            # make predictions on the test set
            y_pred = rf_model.predict(x_test)

            # assess the accuracy
            total_preds = 0
            total_correct = 0

            for j in range(0, y_pred.shape[0]):
                total_preds += 1
                if np.array_equal(y_pred[j], y_test.values[j]):
                    total_correct += 1

            accuracy = total_correct / total_preds
            metrics[f"iteration_{i}_accuracy"] = accuracy

            # determine whether to update parameters
            if accuracy > best_accuracy:
                best_accuracy = accuracy

                best_n_estimators = n_estimators
                best_max_depth = max_depth
                best_min_samples_split = min_samples_split
                best_min_samples_leaf = min_samples_leaf
                best_max_features = max_features

                # create a dictionary with the results
                best_params = {
                    "n_estimators": best_n_estimators,
                    "max_depth": best_max_depth,
                    "min_samples_split": best_min_samples_split,
                    "min_samples_leaf": best_min_samples_leaf,
                    "max_features": best_max_features,
                }
                # write parameters to disk

        with open(best_params_file + ".json", "w") as f:
            json.dump(best_params, f)

        logging.info(
            "Completed hp tuning iteration {}, best accuracy {} with params {}".format(
                str(i + 1), str(best_accuracy), best_params
            )
        )

    else:

        logging.info("Parameters loaded {}".format(str(best_params)))

    # fit a model on the entire training set with the best parameters
    logging.info("Fitting model across whole training set")
    rf_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params["max_features"],
        random_state=15,
        n_jobs=-1,
    )
    rf_model.fit(x_train_df, y_train_df)

    # export the classifier to a file
    logging.info("Exporting model to Cloud Storage")
    model_filename = "model.pkl"
    # dump(rf_model, model_file.path + ".pkl")

    with open(model_file.path + ".pkl", "wb") as f:
        pickle.dump(rf_model, f)

    # assess the model accuracy
    # make predictions on the test set
    logging.info("Assessing model accuracy")
    y_pred = rf_model.predict(x_test_df)
    total_preds = 0
    total_correct = 0
    for i in range(0, y_pred.shape[0]):
        total_preds += 1

        if np.array_equal(y_pred[i], y_test_df.values[i]):
            total_correct += 1

    accuracy = str(round((total_correct / total_preds) * 100))
    metrics["final_model_accuracy"] = accuracy
    logging.info("Predictions correct for {}% of test samples".format(accuracy))

    output = json.dumps(metrics)
    with open(metrics_file + ".json", "w") as f:
        json.dump(metrics, f)
