# Iris Flower Classification

## Project Overview
This project uses Machine Learning to classify Iris flowers into three species: Setosa, Versicolor, and Virginica.

A Random Forest Classifier is trained using the Iris dataset to predict the species of a flower based on its measurements.

## Objectives
- Understand the basics of Machine Learning.
- Explore and analyze the Iris dataset.
- Train a classification model using Python.
- Evaluate the model's performance.
- Predict the species of a new flower.

## Technologies Used
- Python
- Google Colab
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

## Dataset
The Iris dataset contains 150 flower samples with four features:

- Sepal length
- Sepal width
- Petal length
- Petal width

The three target classes are Setosa, Versicolor, and Virginica.

## Project Workflow
1. Import the required Python libraries.
2. Load and explore the Iris dataset.
3. Check for missing values.
4. Visualize flower measurements.
5. Split the dataset into training and testing sets.
6. Train a Random Forest Classifier.
7. Evaluate the model using accuracy and a classification report.
8. Display the confusion matrix.
9. Predict the species of a new flower.
10. Save the trained model using Joblib.

## Model Details
- Algorithm: Random Forest Classifier
- Number of trees: 100
- Training samples: 120
- Testing samples: 30
- Test accuracy: 90%

## Sample Prediction
**Input measurements:**
- Sepal length: 5.1 cm
- Sepal width: 3.5 cm
- Petal length: 1.4 cm
- Petal width: 0.2 cm

**Predicted species:** Setosa

## Project Files
- `CodeAlpha_Iris_Flower_Classification.ipynb` — Notebook containing the code, analysis, model training, and evaluation.
- `iris_model.pkl` — Saved trained machine learning model.

## How to Run
1. Open Google Colab.
2. Upload `CodeAlpha_Iris_Flower_Classification.ipynb`.
3. Run the notebook cells in order.
4. To use the saved model separately, load `iris_model.pkl` with Joblib and provide flower measurements in the expected feature order.

## Conclusion
The project demonstrates a basic machine learning workflow, from dataset exploration to model training, evaluation, and prediction. The trained model achieved 90% accuracy on the selected test dataset.

## Author
CodeAlpha Data Science Internship Project
