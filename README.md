# MSB1015 - Project
This repository contains the code of my project for Scientific Programming (MSB1015) course. The aim of this project is to preprocess data that is used in machine learning algorithms for feature selection.

## Prerequisites
Before running the Jupyter Notebook files, ensure that you have the following installed:
[Python 3.7+](https://www.python.org/downloads/)
**Jupyter Notebook of Jupyter Lab**

## Installation
Clone the git repo:
```bash
git clone https://github.com/MaxVanHout/MSB1015.git
```
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
## Usage
Run the preprocessing script first that takes the uncleaned data as input and outputs clean data:
```bash
python preprocesing.ipynb
```
Run the feature selection script that takes the clean data as imput and runs the ML models:
```bash
python feature_selection.ipynb
```



