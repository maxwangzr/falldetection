# Evidence Detection (ED) System

## Project Overview
This project utilizes a BERT-based deep learning approach to develop an evidence detection system that analyzes the relationship between claims and their corresponding evidence. The model used is a fine-tuned version of the pre-trained BERT model, which is renowned for its effectiveness in various natural language processing tasks including text classification.
The fine-tuning process involves adapting the BERT model to our specific task of evidence detection, enhancing its ability to discern whether evidence supports or contradicts a given claim.

## Dataset
Our model training process involved using `train.csv` for initial training and `dev.csv` for validation to fine-tune hyperparameters. Once we achieved satisfactory performance, we combined these datasets to retrain the models. This retraining step was crucial as it allowed us to enhance the models' predictive capabilities, leading to improved accuracy in our evidence detection system.

## Getting Started

### Prerequisites
- Python 3.6 or higher
- PyTorch
- Transformers library by Hugging Face
- Access to a GPU is recommended for model training and evaluation

### Installation
1. Start your virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

2. Install the required Python packages:
```bash 
pip install transformers torch pandas scikit-learn
```
3. Download pre-trained models from OneDrive or train them using the provided notebooks:
   - [Best Model - BERT for Evidence Detection](https://livemanchesterac-my.sharepoint.com/:u:/g/personal/zhuoran_wang-2_student_manchester_ac_uk/EaSVcTouJQZJgY7RPBB9u2QBPT1g4Fe6FpPSGGRhhPW6_w?e=4Ne7mH)




### Training the Model
To train the ED model, navigate to `TrainEvaluation.ipynb` and run the Notebook after adjusting the file paths accordingly.

### Generating Predictions
After obtaining the models, navigate to `Test.ipynb` and adjust the file paths for 'test_data' to generate predictions.

## Folder Structure Overview
- **data/**: Contains data for training and validating the models. It also stores the prediction files unless specified otherwise.
- **models/**: 'best_model.bin':Stores the trained model checkpoints.
- **scripts/**: Contains Jupyter notebooks for various purposes:
 -  `TrainEvaluation.ipynb`: train and evaluate the model.
 - `Test.ipynb` load the trained model and make predictions on new data

Thank you for exploring the Evidence Detection System project.
