# Credit Card Fraud Detection 

This project implements a Machine Learning model to detect fraudulent credit card transactions. It aims to solve the problem of identifying unauthorized transactions so that customers are not charged for items they did not purchase.

## Project Overview
Credit card fraud detection is a classic **imbalanced classification** problem. The dataset typically contains a very small percentage of fraudulent transactions compared to legitimate ones. This project explores data preprocessing, handling class imbalance, and training various machine learning models to accurately classify transactions.

## Dataset
The dataset used in this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) .

- **Content**: Transactions made by credit cards in September 2013 by European cardholders.
- **Features**: 
  - `Time`: Seconds elapsed between each transaction and the first transaction.
  - `V1` to `V28`: Principal components obtained with PCA (for confidentiality).
  - `Amount`: Transaction amount.
  - `Class`: Target variable (1 for Fraud, 0 for Valid).

## Technologies Used
- **Python**
- **Pandas** & **NumPy** (Data manipulation)
- **Matplotlib** & **Seaborn** (Data visualization)
- **Scikit-Learn** (Machine Learning models & preprocessing)
- **Jupyter Notebook**

## Key Steps
1. **Exploratory Data Analysis (EDA)**: Analyzing the distribution of legit vs. fraud transactions and feature correlations.
2. **Data Preprocessing**: 
   - Scaling the `Amount` and `Time` columns.
   - Handling class imbalance (e.g., using Undersampling or Oversampling/SMOTE).
3. **Model Training**: Implementing algorithms such as:
   - Logistic Regression
   - Random Forest Classifier
   - Decision Trees
4. **Evaluation**: Using metrics like **Precision, Recall, F1-Score**, and the **Confusion Matrix** (since Accuracy is misleading for imbalanced data).

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/Joya-01/Credit-Card-Fraud-Detection.git](https://github.com/Joya-01/Credit-Card-Fraud-Detection.git)

2. Navigate to the project directory:
   ```bash
   cd Credit-Card-Fraud-Detection
3. Install dependencies (if you have a requirements file, otherwise install manually):
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Credit_Card_Fraud_Detection.ipynb

## License
This project is open-source and available under the MIT License.
