
CREATE OR REPLACE MODEL `mwpmltr.mlops_bqml_text_analyisis.logistic_reg`
  OPTIONS (
      model_type='logistic_reg',
      input_label_cols=['label']) AS
  SELECT
      label,
      feature.*
  FROM
     `mwpmltr.mlops_bqml_text_analyisis.reuters_text_preprocessed_0obvksai`
  WHERE split = 'TRAIN';
