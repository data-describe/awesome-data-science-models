# Goal

We will be using the Malaria dataset in this demo.

The Malaria dataset contains a total of 27,558 cell images with equal instances of parasitized and uninfected cells from the thin blood smear slide images of segmented cells.We will be doing classification of malaria-infected red blood cells using deep learning


├── answer
│   ├── AI Platform.ipynb
│   └── trainer
│       ├── __init__.py
│       └── trainer.py
├── EDA
│   └── EDA.ipynb
└── README.md


**answers/AI Platform.ipynb**
<br>
All of the commands necessary to tune, train, and host the model on AI Platform. This is a step-by-step notebook with explicit instructions on how to carry out the tasks outlined in the "Goals" section above.

**answers/trainer/trainer.py**
<br>
Main module responsible for accepting input parameters and executing the training job.

**EDA/EDA.ipynb**
<br>
Notebook which runs an Exploratory Data Analysis on the dataset.