# TASK -1 Iris Flower Classification
This repository contains the code for training a machine learning model to classify Iris flowers based on their measurements. The Iris flower dataset consists of three species: Setosa, Versicolor, and Virginica, each having distinct measurement characteristics. The goal is to develop a model that can accurately classify Iris flowers based on their measurements.

![Iris species image](https://github.com/AnishaBeera/OIBSIP-Oasis-Infobyte-task1/assets/171479100/4c521e1d-1a21-460b-a216-a602eef926bc)

## Introduction
The Iris flower dataset is a classic dataset used in machine learning and statistics. It includes measurements of sepal length, sepal width, petal length, and petal width for 150 Iris flowers, with 50 samples for each species. The objective is to build models that can classify Iris flowers into their respective species based on these measurements.

## Dataset
The dataset used for this project is the famous Iris flower dataset. It includes:

Sepal length
Sepal width
Petal length
Petal width
Species (target variable)
The dataset is available in this repository as iris.csv.

## Dependencies
The following Python libraries are used in this project:

NumPy
Pandas
Seaborn
Matplotlib
scikit-learn

## Model Training
The code for training the classification models can be found in the iris_classification.ipynb Jupyter Notebook. Three different models are trained:

Support Vector Machine (SVM)
Logistic Regression
Decision Tree Classifier
Each model is trained using the Iris flower dataset and evaluated for its accuracy.

## Testing
After training the models, a new test dataset is used to assess their performance. The test dataset contains measurements of Iris flowers with unknown species. The trained models predict the species of these flowers, and their accuracy is evaluated.

## Installation
You can install the required libraries using pip:

!pip install numpy
!pip install pandas
!pip install scikit-learn
!pip install matplotlib

## Results
The performance of the models is visualized using various plots. Below is an example of visualization created during the project:

## Pair Plot
This pair plot visualizes the relationships between different features of the Iris dataset, colored by species.


## Confusion Matrix
The confusion matrix below shows the performance of the Decision Tree Classifier on the test set.


## Model Accuracy
The accuracy scores of the three models are compared in the bar chart below.

## Conclusion
In this project, we successfully demonstrated the classification of Iris flowers into three species—Setosa, Versicolor, and Virginica—using their sepal and petal measurements. By utilizing popular Python libraries such as NumPy, Pandas, Seaborn, Matplotlib, and scikit-learn, we were able to explore the data, visualize important relationships, and build predictive models.

This project serves as a solid foundation for anyone looking to understand the basics of classification tasks using machine learning. It showcases the entire workflow from data preprocessing to model evaluation and highlights the importance of visualizing data and results for better interpretation.

Experimentation with different models and feature engineering techniques can be used to improve classification accuracy even further. Contributions and suggestions for improvements are always welcome.

Thank you for exploring this project, and we hope it provides a useful resource for your machine learning journey!
