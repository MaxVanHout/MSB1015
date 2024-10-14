# MSB1015 - Project
This repository contains the code of my project for Scientific Programming (MSB1015) course. The aim of this project is to preprocess data that is used in machine learning algorithms for feature selection.

## Prerequisites
Before running the Jupyter Notebook files, ensure that you have the following installed:

[Python 3.7+](https://www.python.org/downloads/)

**Jupyter Lab of Jupyter Notebook**: 
```bash
pip install jupyterlab
```
```bash
pip install notebook
```
and run with the following command
```bash
jupyter lab
```
```bash
jupyter notebook
```
## Installation
Clone the git repo:
```bash
git clone https://github.com/MaxVanHout/MSB1015.git
cd MSB1015
```
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
## Usage
1) Run the **preprocessing.ipynb** script first: this takes the uncleaned **data.csv** file as input and outputs the cleaned **data_cleaned.csv** file to be used in further analysis.

2) Run the **feature_selection.ipynb** script next: this takes the **data_cleaned.csv** file as imput and performs various feature selection algorithms.

3) The **functions.py** file contains functions required to run the other two files.

## Contact
Max Van Hout - maxvanhout00@gmail.com

Project Link: https://github.com/MaxVanHout/MSB1015




