#!/usr/bin/env python3

import kfp
from kfp import dsl
import kfp.gcp as gcp


def collect_stats_op(): #symbol
    return dsl.ContainerOp(
        name='Collect Stats',
        image='gcr.io/ross-kubeflow/collect-stats:latest'       
        
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))


def feature_eng_op(): 
    return dsl.ContainerOp(
        name='Feature Engineering',
        image='gcr.io/ross-kubeflow/feature-eng:latest',   
        
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))


def train_test_val_op(pitch_type): 
    return dsl.ContainerOp(
        name='Split Train Test Val',
        image='gcr.io/ross-kubeflow/train-test-val:latest',
        arguments=[
            '--pitch_type', pitch_type
        ]    
        
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))


def tune_hp_op(pitch_type): 
    return dsl.ContainerOp(
        name='Tune Hyperparameters',
        image='gcr.io/ross-kubeflow/tune-hp:latest',
        arguments=[
            '--pitch_type', pitch_type
        ]    
        
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))


def train_xgboost_op(pitch_type): 
    return dsl.ContainerOp(
        name='Train XGBoost',
        image='gcr.io/ross-kubeflow/train-xgboost:latest',
        arguments=[
            '--pitch_type', pitch_type
        ]    
        
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))


def host_xgboost_op(pitch_type): 
    return dsl.ContainerOp(
        name='Host Model',
        image='gcr.io/ross-kubeflow/host-xgboost:latest',
        arguments=[
            '--pitch_type', pitch_type
        ] 
        
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))


def find_threshold_op(pitch_type): 
    return dsl.ContainerOp(
        name='Find Threshold',
        image='gcr.io/ross-kubeflow/find-threshold:latest',
        arguments=[
            '--pitch_type', pitch_type
        ]    
        
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))


def evaluate_model_op(pitch_type, dummy1=None): 
    return dsl.ContainerOp(
        name='Evaluate Models',
        image='gcr.io/ross-kubeflow/evaluate-model:latest',
        arguments=[
            '--pitch_type', pitch_type
        ],
        file_outputs={
            'data': '/root/dummy.txt',
        } 
        
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))


def enhance_features_op(dummy_1=None, dummy_2=None, dummy_3=None, dummy_4=None, dummy_5=None, dummy_6=None, dummy_7=None, dummy_8=None, dummy_9=None, dummy_10=None, dummy_11=None, dummy_12=None): 
    return dsl.ContainerOp(
        name='Enhance Features',
        image='gcr.io/ross-kubeflow/enhance-features:latest',
        arguments=[
            '--dummy_1', dummy_1,
            '--dummy_2', dummy_2,
            '--dummy_3', dummy_3,
            '--dummy_4', dummy_4,
            '--dummy_5', dummy_5,
            '--dummy_6', dummy_6,
            '--dummy_7', dummy_7,
            '--dummy_8', dummy_8,
            '--dummy_9', dummy_9,
            '--dummy_10', dummy_10,
            '--dummy_11', dummy_11,
            '--dummy_12', dummy_12
        ]
        
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))


def train_rf_op(): #symbol
    return dsl.ContainerOp(
        name='Train RF',
        image='gcr.io/ross-kubeflow/train-rf:latest'       
        
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))


def host_rf_op(): #symbol
    return dsl.ContainerOp(
        name='Host RF',
        image='gcr.io/ross-kubeflow/host-rf:latest'       
        
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))


@dsl.pipeline(
    name='Sequential pipeline',
    description='A pipeline with sequential steps.'
)
def sequential_pipeline():  
    """A pipeline with sequential steps.""" 
    
    refresh_data_pipeline = feature_eng_op().after(collect_stats_op())

    FT_task = evaluate_model_op('FT').after(find_threshold_op('FT').after(host_xgboost_op('FT').after(train_xgboost_op('FT').after(tune_hp_op('FT').after(train_test_val_op('FT').after(refresh_data_pipeline)))))) 
    FS_task = evaluate_model_op('FS').after(find_threshold_op('FS').after(host_xgboost_op('FS').after(train_xgboost_op('FS').after(tune_hp_op('FS').after(train_test_val_op('FS').after(refresh_data_pipeline)))))) 
    CH_task = evaluate_model_op('CH').after(find_threshold_op('CH').after(host_xgboost_op('CH').after(train_xgboost_op('CH').after(tune_hp_op('CH').after(train_test_val_op('CH').after(refresh_data_pipeline))))))
    FF_task = evaluate_model_op('FF').after(find_threshold_op('FF').after(host_xgboost_op('FF').after(train_xgboost_op('FF').after(tune_hp_op('FF').after(train_test_val_op('FF').after(refresh_data_pipeline))))))
    SL_task = evaluate_model_op('SL').after(find_threshold_op('SL').after(host_xgboost_op('SL').after(train_xgboost_op('SL').after(tune_hp_op('SL').after(train_test_val_op('SL').after(refresh_data_pipeline))))))
    CU_task = evaluate_model_op('CU').after(find_threshold_op('CU').after(host_xgboost_op('CU').after(train_xgboost_op('CU').after(tune_hp_op('CU').after(train_test_val_op('CU').after(refresh_data_pipeline))))))
    FC_task = evaluate_model_op('FC').after(find_threshold_op('FC').after(host_xgboost_op('FC').after(train_xgboost_op('FC').after(tune_hp_op('FC').after(train_test_val_op('FC').after(refresh_data_pipeline))))))
    SI_task = evaluate_model_op('SI').after(find_threshold_op('SI').after(host_xgboost_op('SI').after(train_xgboost_op('SI').after(tune_hp_op('SI').after(train_test_val_op('SI').after(refresh_data_pipeline))))))
    KC_task = evaluate_model_op('KC').after(find_threshold_op('KC').after(host_xgboost_op('KC').after(train_xgboost_op('KC').after(tune_hp_op('KC').after(train_test_val_op('KC').after(refresh_data_pipeline)))))) 
    EP_task = evaluate_model_op('EP').after(find_threshold_op('EP').after(host_xgboost_op('EP').after(train_xgboost_op('EP').after(tune_hp_op('EP').after(train_test_val_op('EP').after(refresh_data_pipeline))))))
    KN_task = evaluate_model_op('KN').after(find_threshold_op('KN').after(host_xgboost_op('KN').after(train_xgboost_op('KN').after(tune_hp_op('KN').after(train_test_val_op('KN').after(refresh_data_pipeline))))))
    FO_task = evaluate_model_op('FO').after(find_threshold_op('FO').after(host_xgboost_op('FO').after(train_xgboost_op('FO').after(tune_hp_op('FO').after(train_test_val_op('FO').after(refresh_data_pipeline))))))

    enhance_features_task = enhance_features_op(FT_task.output,FS_task.output,CH_task.output,FF_task.output,SL_task.output,CU_task.output,FC_task.output,SI_task.output,KC_task.output,EP_task.output,KN_task.output,FO_task.output)

    rf_ensemble_task = host_rf_op().after(train_rf_op().after(enhance_features_task))
    

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(sequential_pipeline, __file__ + '.zip')