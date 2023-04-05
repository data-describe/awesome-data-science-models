import argparse
import datetime
import pandas as pd
import subprocess
import pickle
from google.cloud import storage
import hypertune
import xgboost as xgb


# Load the hyperparameter values that are passed to the model during training.
parser = argparse.ArgumentParser()

parser.add_argument(
    "--job-dir",  # handled automatically by AI Platform
    help="GCS location to write checkpoints and export models",
    required=True,
)
parser.add_argument(
    "--max_depth",  # Specified in the config file
    help="Maximum depth of the XGBoost tree. default: 3",
    default=3,
    type=int,
)
parser.add_argument(
    "--num_boost_round",  # Specified in the config file
    help="Number of boosting iterations. default: 100",
    default=100,
    type=int,
)
parser.add_argument(
    "--booster",  # Specified in the config file
    help="which booster to use: gbtree, gblinear or dart. default: gbtree",
    default="gbtree",
    type=str,
)

parser.add_argument(
    "--project-id",
    dest='project_id',
    type=str,
    default="mwe-sanofi-ml-workshop",
    help="The GCP Project ID",
)
parser.add_argument(
    "--bucket-name",
    dest='bucket_name',
    type=str,
    default="ml-lending-club-demo",
    help="The Cloud Storage bucket to be used for process artifacts",
)

args = parser.parse_args()

# Add code to download the data from GCS

# Bucket holding the lending club data
bucket = storage.Client().bucket(args.bucket_name)
# Path to the data inside the bucket
blob = bucket.blob("lending_club/data/lending_club_data.tsv")
# Download the data
blob.download_to_filename("lending_club_data.tsv")

# Return first three digits of zip code.
z3 = lambda z: int(z[:3])


def dateparse(dt):
    # test for empty string '' or NaT
    if not dt or pd.isnull(dt):
        return pd.NaT
    m, d, y = map(int, dt.split("/"))
    # 68 ==> 1968, 07 ==> 2007
    y = y + 1900 if y > 20 else y + 2000
    return pd.datetime(y, m, d)


schema = pd.Series(
    {
        "Id": "int64",
        "is_bad": "int64",
        # 'emp_title': 'category',
        "emp_length": "float64",
        "home_ownership": "object",
        "annual_inc": "float64",
        "verification_status": "object",
        # 'pymnt_plan': 'category',
        # 'Notes': 'category',
        "purpose_cat": "object",
        # 'purpose': 'category',
        "zip_code": "int64",
        "addr_state": "object",
        "debt_to_income": "float64",
        "delinq_2yrs": "float64",
        "earliest_cr_line": "datetime64[ns]",
        "inq_last_6mths": "float64",
        "mths_since_last_delinq": "float64",
        "mths_since_last_record": "float64",
        "open_acc": "float64",
        "pub_rec": "float64",
        "revol_bal": "float64",
        "revol_util": "float64",
        "total_acc": "float64",
        # 'initial_list_status': 'category',
        # 'collections_12_mths_ex_med': 'Int64',
        "mths_since_last_major_derog": "int64",
        "policy_code": "object",
    }
)

usecols = schema.index
# zip_code and earliest_cr_line will be parsed through converters.
dtype = schema[~schema.index.isin(["zip_code", "earliest_cr_line"])].to_dict()

lcd = pd.read_csv(
    "./lending_club_data.tsv",
    sep="\t",
    index_col="Id",
    na_values={"emp_length": "na"},
    keep_default_na=True,
    usecols=usecols,
    dtype=dtype,
    converters={"zip_code": z3, "earliest_cr_line": dateparse},
)

lcd["emp_length"].fillna(1, inplace=True)
lcd["emp_length"].clip(upper=10, inplace=True)
lcd["emp_length"] = lcd["emp_length"].astype("int64")
# lcd['emp_length'].value_counts(dropna=False)


# lcd['zip_code'].dtype
# dtype('int64')

lcd["home_ownership"].replace("NONE", "OTHER", inplace=True)
# lcd['home_ownership'].value_counts(dropna=False)

# lcd['annual_inc'].median()
lcd["annual_inc"].fillna(58000, inplace=True)
lcd["annual_inc"].clip(upper=250000, inplace=True)
# lcd['annual_inc'].value_counts(dropna=False)

sb_flag = lcd["purpose_cat"].str[-14:] == "small business"
lcd.loc[sb_flag, "purpose_cat"] = "small business"
# np.where(sb_flag, 'small business', lcd['purpose_cat'])
# lcd['purpose_cat'].value_counts(dropna=False)


lcd["delinq_2yrs"].fillna(0, inplace=True)
lcd["delinq_2yrs"].clip(upper=3, inplace=True)
lcd["delinq_2yrs"] = lcd["delinq_2yrs"].astype("int64")
# lcd['delinq_2yrs'].value_counts(dropna=False)

lcd["inq_last_6mths"].fillna(0, inplace=True)
lcd["inq_last_6mths"].clip(upper=4, inplace=True)
lcd["inq_last_6mths"] = lcd["inq_last_6mths"].astype("int64")
# lcd['inq_last_6mths'].value_counts(dropna=False)
# lcd.groupby('inq_last_6mths', observed=True).agg({'is_bad': ['sum', 'count']})

lcd["mths_since_last_delinq"].fillna(120.0, inplace=True)
# lcd.groupby('mths_since_last_delinq', observed=True).agg({'is_bad': ['sum', 'count']})

lcd["mths_since_last_record"].fillna(0.0, inplace=True)
lcd["inq_last_6mths"] = lcd["inq_last_6mths"].astype("int64")
# lcd.groupby('mths_since_last_record', observed=True).agg({'is_bad': ['sum', 'count']})

lcd["open_acc"].fillna(7, inplace=True)
lcd["open_acc"] = lcd["open_acc"].astype("int64")
# lcd['open_acc'].value_counts(dropna=False)
# lcd.groupby('open_acc', observed=True).agg({'is_bad': ['sum', 'count']})

lcd["pub_rec"].fillna(0, inplace=True)
lcd["pub_rec"].clip(upper=1, inplace=True)
lcd["pub_rec"] = lcd["pub_rec"].astype("int64")
# lcd['pub_rec'].value_counts(dropna=False)
# lcd.groupby('pub_rec', observed=True).agg({'is_bad': ['sum', 'count']})

lcd["revol_bal"].clip(upper=100000, inplace=True)
# lcd.groupby('revol_bal', observed=True).agg({'is_bad': ['sum', 'count']})

lcd["revol_util"].fillna(0, inplace=True)
# lcd.groupby('revol_util', observed=True).agg({'is_bad': ['sum', 'count']})

lcd["total_acc"].fillna(1, inplace=True)
lcd["total_acc"] = lcd["total_acc"].astype("int64")
# lcd['total_acc'].value_counts(dropna=False)
# lcd.groupby('total_acc', observed=True).agg({'is_bad': ['sum', 'count']})

# lcd.groupby('collections_12_mths_ex_med').agg({'is_bad': ['sum', 'count']})
target = "is_bad"
numerical_features = lcd.select_dtypes(include=["number"]).columns
numerical_features = numerical_features.drop([target])

categorical_features = lcd.select_dtypes(include=["object"]).columns
predictors = numerical_features.union(categorical_features, sort=False)

# exec(open("/mnt/c/Users/bjaco/Documents/projects_2020/mavenwave/python/lcd_1/lcd_read_input_1.py").read())
# ecl=pd.concat([lcd['earliest_cr_line'], lcd[target]], axis=1, copy=True)
# ecl.set_index('earliest_cr_line', inplace=True)
# ecl_q = ecl.groupby(by=pd.Grouper(freq='Q')).agg({target: ['sum', 'count']})
# pd.set_option('display.max_rows', 100)
# print(ecl_q[-100:])

#                  is_bad
#                     sum count
# earliest_cr_line
# 2006-09-30           10    87
# 2006-12-31           16    82
# 2007-03-31           12    64
# 2007-06-30           11    53
# 2007-09-30            7    39
# 2007-12-31            3    24
# 2008-03-31            1    21
# 2008-06-30            2     9
# 2008-09-30            0     3
# 2008-12-31            0     1

lcd.reset_index("Id", inplace=True)
lcd.set_index("earliest_cr_line", inplace=True)
start_train = "2002-7-1"
start_test = "2006-7-1"
end_test = "2007-7-1"

lcd_train = lcd.loc[start_train:start_test].copy()
lcd_test = lcd.loc[start_test:end_test].copy()


lcd_train.reset_index("earliest_cr_line", drop=True, inplace=True)
lcd_test.reset_index("earliest_cr_line", drop=True, inplace=True)

# lcd.set_index(['Id'], append=True, inplace=True)
lcd_train.set_index(["Id"], inplace=True)
lcd_test.set_index(["Id"], inplace=True)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

X_train = lcd_train[predictors].copy()
y_train = lcd_train[target].copy()
# X_train.head()

X_test = lcd_test[predictors].copy()
y_test = lcd_test[target].copy()
# Manually one-hot-encode categorical variables.
# https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
for cat_var in list(categorical_features.values):
    cat_col = pd.DataFrame(X_train[cat_var], columns=[cat_var])
    dum_df = pd.get_dummies(cat_col, columns=[cat_var], prefix=cat_var)
    X_train.drop(cat_var, axis=1, inplace=True)
    X_train = X_train.join(dum_df)
# X_train.columns

for cat_var in list(categorical_features.values):
    cat_col = pd.DataFrame(X_test[cat_var], columns=[cat_var])
    dum_df = pd.get_dummies(cat_col, columns=[cat_var], prefix=cat_var)
    X_test.drop(cat_var, axis=1, inplace=True)
    X_test = X_test.join(dum_df)
# X_test.columns
# set value of zero for columns which are in training set but not in test set (result of transforming categorical column to set of dummy columns)

not_in_test = [ item for item in X_train.columns.tolist() if item not in X_test.columns.tolist() ]
if len(not_in_test) >1 :
    for col in not_in_test :
        X_test[col]=0      
# remove columns which are not in training set
X_test = X_test[X_train.columns.tolist()]

# Use the Hyperparameters

# Create the classifier, here we will use an XGboost classifier to demonstrate the use of HP Tuning.
# Here is where we set the variables used during HP Tuning from
# the parameters passed into the python script
# perform    transformation  


classifier = xgb.XGBClassifier(
    max_depth=args.max_depth,
    num_boost_round=args.num_boost_round,
    booster=args.booster,
    eval_metric="auc",
)

# Transform the features and fit them to the classifier
# classifier.fit(train_df[FEATURES], train_df[TARGET])
classifier.fit(X_train, y_train)

# Report the mean accuracy as hyperparameter tuning objective metric.
# Calculate the mean accuracy on the given test data and labels.
score = classifier.score(X_test, y_test)


# The default name of the metric is training/hptuning/metric.
# We recommend that you assign a custom name. The only functional difference is that
# if you use a custom name, you must set the hyperparameterMetricTag value in the
# HyperparameterSpec object in your job request to match your chosen name.
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#HyperparameterSpec
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag="rmseORmlogloss", metric_value=score, global_step=100
)

# Export and save the model to GCS
# Export the model to a file
now = (datetime.datetime.now() + datetime.timedelta(hours=-5)).strftime("%Y%m%d_%H%M%S") # Central Time
model_filename = 'model_{}.pkl'.format(now)
with open(model_filename, "wb") as f:
    pickle.dump(classifier, f)

# upload the saved model file to Cloud Storage
subprocess.check_call(
    [
        "gsutil",
        "cp",
        model_filename,
        "gs://" + str(args.bucket_name) + "/lending_club/model/",
    ]
)


# upload the test data to Cloud Storage.for
test_data_filename = 'test_set.csv'
X_test.to_csv('test_set.csv')
subprocess.check_call(
    [
        "gsutil",
        "cp",
        test_data_filename,
        "gs://" + str(args.bucket_name) + "/lending_club/data/",
    ]
)


# # Example: job_dir = 'gs://BUCKET_ID/xgboost_job_dir/1'
# job_dir =  args.job_dir.replace('gs://', '')  # Remove the 'gs://'
# # Get the Bucket Id
# bucket_id = job_dir.split('/')[0]
# # Get the path
# bucket_path = job_dir[len('{}/'.format(bucket_id)):]  # Example: 'xgboost_job_dir/1'
# # (bucket_id, _, bucket_path) = job_dir.partition('/')

# # Upload the model to GCS
# bucket = storage.Client().bucket(bucket_id)
# blob = bucket.blob('{}/{}'.format(
#     bucket_path,
#     model_filename))

# blob.upload_from_filename(model_filename)
