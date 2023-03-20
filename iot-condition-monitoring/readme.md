# IoT Condition Monitoring


## Use Case

Many plants faces unplanned downtime and it assets fail over the period of time. With the help of data science, the main aim of this project is to provide plants an ability to to analyze the condition or health of machine assets using a ML tool. 

## Paderborn Bearing Dataset 

### Experiment Setup

The basic components of the load test bench are the drive motor acting as a sensor, a torque measuring shaft, the test modules and a load motor. Ball or cylindrical roller bearings of type 6203, N203 or NU203 are used in the experiment. The main measured variables for generating the database are the stator current on the drive motor and the housing vibrations in the form of accelerations on the bearing housing in the rolling bearing test module.

![alt text](https://github.com/arivperumal19/awesome-data-science-models/blob/condition-monitoring/iot-condition-monitoring/images/experiment_setup.png)

### Operating Conditions

The Experiment was conducted under 4 different operating conditions to ensure the robustness of condition monitoring methods.

![alt text](https://github.com/arivperumal19/awesome-data-science-models/blob/condition-monitoring/iot-condition-monitoring/images/operating_conditions.png)


Our dataset of Interest(Paderborn Bearing Dataset) consists the following details:

1. The Dataset contains the motor currents and vibration signals with additional measurements of torque, speed, load and temperature
2. There are 26 Damaged bearing and 6 healthy bearing state in the data
3. All the data is collected in 4 different operating conditions.

Dataset - https://mb.uni-paderborn.de/en/kat/main-research/datacenter/bearing-datacenter/data-sets-and-download

## Goal

With help of Data Science techniques the aim is to analyze the condition of bearings, identify features which contribute to bearing damage and have a  classification model to predict the extent of the damage.
