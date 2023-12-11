# Image Classification Using VGG16 Model
## Project Overview
This project involves developing an image classification system using the VGG16 model, a well-known architecture in the field of deep learning. The objective is to accurately classify images into four predefined categories, showcasing the model's ability to learn and predict from visual data.

## Key Features
Customized VGG16 Model: Leveraged the VGG16 model with modifications to the classifier to suit our specific classification needs.
Data Augmentation: Implemented several image transformation techniques to enhance model generalization.
Dynamic Learning Rate Adjustment: Utilized a learning rate scheduler to optimize the training process.
Performance Evaluation: Employed accuracy metrics and a confusion matrix for comprehensive model evaluation.

## Technologies Used
Python
PyTorch
torchvision
PIL (Python Imaging Library)
NumPy
Seaborn
Matplotlib
scikit-learn

## Setup and Usage

1. Environment Setup:

Ensure Python 3.x is installed.
Install required libraries: pip install -r requirements.txt.

2. Training the Model:

Run python model/train.py to start the training process.
Model checkpoints and logs will be saved in trained_models/.

3. Evaluating the Model:

Evaluate the trained model using python model/evaluate.py.

4. Running Predictions:

Use python model/predict.py to run predictions on new images.

## Results and Discussion
Train Accuracy= 100%, 
Validation Accuracy=90.77%
