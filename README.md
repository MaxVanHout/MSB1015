# MSB1015 - Project
This repository contains the code of my project for the Scientific Programming (MSB1015) course. The aim of this project is to preprocess data that is used in machine learning algorithms for feature selection.

The [Toxicity dataset](https://archive.ics.uci.edu/dataset/728/toxicity-2) used for this project includes 171 molecules designed for functional domains of a core clock protein, CRY1, responsible for generating circadian rhythm. 56 of the molecules are toxic and the rest are non-toxic.


## Prerequisites
Before running the Jupyter Notebook files, ensure that you have the following installed:

[PIP Package Manager](https://pypi.org/project/pip/)

[Python 3.7+](https://www.python.org/downloads/)

**Jupyter Lab of Jupyter Notebook**: 
```bash
pip install jupyterlab
```
```bash
pip install notebook
```
and run with the following command:
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
```
Use [pip](https://pip.pypa.io/en/stable/) to install the required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage
1) Run the **preprocessing.ipynb** script first: this takes the uncleaned **data.csv** file as input and outputs the cleaned **data_cleaned.csv** file to be used in further analysis.

2) Run the **feature_selection.ipynb** script next: this takes the **data_cleaned.csv** file as imput and performs various feature selection algorithms.

3) The **functions.py** file contains functions required to run the other two files.


## License 
See the [LICENSE](LICENSE.md) file for license rights and limitations [(MIT).](https://choosealicense.com/licenses/mit/)


## Contact
Max Van Hout - maxvanhout00@gmail.com

Project Link: https://github.com/MaxVanHout/MSB1015




