# 9517_25T1_4356

## Project Overview
This project implements various deep learning models for image classification tasks, including ResNet, Vision Transformer (ViT), and Kolmogorov-Arnold Networks (KAN). The implementation includes data preparation, feature extraction, and model training pipelines.

## Project Structure
- `Data_Preparation.ipynb`: Notebook for data preprocessing and preparation
- `Feature_Extraction+ML.ipynb`: Feature extraction and machine learning pipeline
- `Resnet18+KAN.ipynb`: Implementation of ResNet18 and KAN models
- `ResNet34+ResNet50.ipynb`: Implementation of ResNet34 and ResNet50 models
- `ViT.ipynb`: Vision Transformer implementation
- `ViT_data_eval.ipynb`: Evaluation of Vision Transformer model

## Dependencies
The project requires the following Python packages:
- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- matplotlib
- jupyter

## Installation
1. Clone this repository:
```bash
git clone https://github.com/boring180/9517_25T1_4356
cd 9517_25T1_4356
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Install pykan:
```bash
git clone https://github.com/KindXiaoming/pykan.git
cd pykan
pip install -e .
```

## Usage
1. Download the dataset from the following link:
[Aerial Landscape Images](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)

2. Start Jupyter Notebook:
```bash
jupyter notebook
```

3. Open the notebooks in the following order:
   - First run `Data_Preparation.ipynb` to prepare your dataset
   - Choose and run one of the model notebooks:
     - `Resnet18+KAN.ipynb`
     - `ResNet34+ResNet50.ipynb`
     - `ViT.ipynb`
     - `Feature_Extraction+ML.ipynb`
