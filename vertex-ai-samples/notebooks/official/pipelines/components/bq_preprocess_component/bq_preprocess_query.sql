
-- create the embedding model
CREATE OR REPLACE MODEL
  `mwpmltr.mlops_bqml_text_analyisis.swivel_text_embedding_model` OPTIONS(model_type='tensorflow',
    model_path='gs://tfhub-modules/google/tf2-preview/gnews-swivel-20dim/1/uncompressed/*');

-- create the preprocessed table
CREATE OR REPLACE TABLE `mwpmltr.mlops_bqml_text_analyisis.reuters_text_preprocessed_0obvksai`
AS (
  WITH
    -- Apply the model for embedding generation
    get_embeddings AS (
      SELECT
        title,
        sentences,
        output_0 as content_embeddings,
        topics
      FROM ML.PREDICT(MODEL `mwpmltr.mlops_bqml_text_analyisis.swivel_text_embedding_model`,(
        SELECT topics, title, content AS sentences
        FROM `mwpmltr.mlops_bqml_text_analyisis.reuters_ingested`
      ))),
    -- Get label
    get_label AS (
        SELECT
            *,
            STRUCT( CASE WHEN 'acq' in UNNEST(topics) THEN 1 ELSE 0 END as acq ) AS label,
        FROM get_embeddings
    ),
    -- Train-serve splitting
    get_split AS (
        SELECT
            *,
            CASE WHEN ABS(MOD(FARM_FINGERPRINT(title), 10)) < 8 THEN 'TRAIN' ELSE 'PREDICT' END AS split
        FROM get_label
    )
    -- create training table
    SELECT
        title,
        sentences,
        STRUCT( content_embeddings[OFFSET(0)] AS content_embed_0,
                content_embeddings[OFFSET(1)] AS content_embed_1,
                content_embeddings[OFFSET(2)] AS content_embed_2,
                content_embeddings[OFFSET(3)] AS content_embed_3,
                content_embeddings[OFFSET(4)] AS content_embed_4,
                content_embeddings[OFFSET(5)] AS content_embed_5,
                content_embeddings[OFFSET(6)] AS content_embed_6,
                content_embeddings[OFFSET(7)] AS content_embed_7,
                content_embeddings[OFFSET(8)] AS content_embed_8,
                content_embeddings[OFFSET(9)] AS content_embed_9,
                content_embeddings[OFFSET(10)] AS content_embed_10,
                content_embeddings[OFFSET(11)] AS content_embed_11,
                content_embeddings[OFFSET(12)] AS content_embed_12,
                content_embeddings[OFFSET(13)] AS content_embed_13,
                content_embeddings[OFFSET(14)] AS content_embed_14,
                content_embeddings[OFFSET(15)] AS content_embed_15,
                content_embeddings[OFFSET(16)] AS content_embed_16,
                content_embeddings[OFFSET(17)] AS content_embed_17,
                content_embeddings[OFFSET(18)] AS content_embed_18,
                content_embeddings[OFFSET(19)] AS content_embed_19) AS feature,
        label.acq as label,
        split
    FROM
      get_split)
