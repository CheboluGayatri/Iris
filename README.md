# Iris Flower Classification

This project predicts the species of an iris flower using measurements of its sepals and petals. The dataset contains three species: Setosa, Versicolor, and Virginica.

# Objective

Classify iris flowers based on four numeric features:

Sepal length

Sepal width

Petal length

Petal width

# Dataset

The project uses the Iris dataset from scikit-learn. It includes 150 samples with no missing values.

Steps

Load and explore the dataset.

Visualize feature patterns with scatter plots and pair plots.

Split the data into training and test sets.

Train models such as Logistic Regression, KNN, or Decision Tree.

Evaluate accuracy and review the confusion matrix.

# Streamlit App

The project includes a Streamlit interface where users can enter measurements and get the predicted iris species. This makes the model easy to use without running Python scripts manually.

Run the app:

streamlit run app.py

Skills Learned

Data exploration and visualization

Building and evaluating classification models

Creating interactive ML apps with Streamlit

End-to-end workflow from dataset to UI

How to Run

Install required libraries:

pip install numpy pandas matplotlib seaborn scikit-learn streamlit


Run the main script:

python iris.py


Run the Streamlit app:

streamlit run app.py --server.port 8999
