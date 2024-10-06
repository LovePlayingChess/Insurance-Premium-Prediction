# Insurance Premium Prediction
This repository contains an end-to-end machine learning project developed for **iNeuron.ai**, aimed at predicting health insurance premiums. The model is built using quantitative features such as age, BMI, and the number of children, alongside categorical features like sex, smoking status, and region. The project showcases the integration of data pipelines, deployment, and real-world machine learning practices.

## Project Overview
This project predicts health insurance expenses using supervised learning techniques. The goal is to estimate an individual's insurance premium based on categorical and quantitative features. The project involves:

* Data preprocessing and feature engineering
* Model building and evaluation
* Deployment of a web application using Flask
* Hosting the application on Microsoft Azure
  
## Key Features
* **Machine Learning Pipeline:**
End-to-end ML workflow, from data ingestion to model deployment.
* **Flask Web Application:**
Model is integrated into a web-based interface, allowing users to input data and receive real-time predictions.
* **Cloud Deployment:**
Deployed on Azure, enabling scalable and remote access.
* **Data Ingestion:**
Data is sourced from multiple locations, including local files, MongoDB, and Cassandra databases.
* **Model Persistence:**
Trained models are saved as .pkl files for future use.
* **Logging:**
Application logs were maintained to track operations and for debugging.
* **Custom Exception Handling:**
Custom exception handling used for efficient debugging.

## Technologies Used
* Programming Languages: Python
* Libraries/Frameworks:
* Flask (for web development)
* Scikit-learn (for machine learning)
* Pandas, NumPy (for data manipulation)
* Matplotlib, Seaborn (for data visualization)
* Database: MongoDB, Cassandra (for data storage)
* Deployment: Microsoft Azure
* Version Control: Git, GitHub
* Model Persistence: Pickle (.pkl files)
* Logging: Python’s built-in logging module

## How to Run the Project Locally
### Prerequisites
Install Python (version 3.8 or higher).
Clone the repository:

**Copy code**
1. git clone https://github.com/username/Insurance-Premium-Prediction.git
2. cd Insurance-Premium-Prediction

Install the required Python packages using:

**Copy code**
1. pip install -r requirements.txt

To run the Flask App locally:

**Copy code**
1. python app.py
2. Open your browser and navigate to http://127.0.0.1:5000/ to interact with the application.

## Acknowledgments
I would like to extend my gratitude to **iNeuron.ai** for providing the opportunity to work on this project. Special thanks to the open-source community for providing useful resources that contributed to the success of this project.
