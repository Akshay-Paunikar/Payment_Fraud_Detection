import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: %(message)s')

project_name = "payment_fraud_detection"

list_of_files = [
    "setup.py",
    "requirements.txt",
    f"{project_name}/exception.py",
    f"{project_name}/logger.py",
    f"{project_name}/utils.py",
    f"{project_name}/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/application/app.py",
    f"{project_name}/application/templates/index.html",
    f"{project_name}/application/templates/static/css/style.css",
    f"{project_name}/application/static/css/style.css",
    f"notebook/payment_fraud_detection_EDA.ipynb",
    f"notebook/payment_fraud_detection_ModelTrainer.ipynb" ,
    f"notebook/datasets/Payment_Fraud.csv"   
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
            
    else:
        logging.info(f"{filename} already exists")