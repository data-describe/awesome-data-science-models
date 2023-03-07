# kubeflow-mlb
kubeflow pipeline for modeling around pitch type

![Image of Pipeline](
https://raw.githubusercontent.com/jonrossclaytor/kubeflow-mlb/master/pipeline-enhance.png)

Original source of Demo: https://github.com/jonrossclaytor/kubeflow-mlb

## Notes
1. The web-scraping component `collect-stats` has been commented out. Consider pulling from data-describe's public datasets (pitch-prediction) instead.
2. The pipeline files have been templated using jinja. Run `python jinja_replace.py <PROJECT_ID>` to apply the files to run on your GCP Project.