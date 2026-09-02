Dataset
The dataset contains 1,540 customer support tickets with 4 columns:
Column	Description
ticket_id	Unique ID of the ticket
ticket	Customer support ticket text
category	Category of the support ticket
priority	Priority of the ticket

The dataset contains the following categories:
Payment Issue
Technical Issue
Order Issue
Subscription
Delivery
Refund
Account Issue

Each category contains 220 tickets.

Priority Distribution

The dataset contains three priority levels:

Medium: 631
High: 606
Low: 303
Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Natural Language Processing
TF-IDF
Logistic Regression
Random Forest Classifier
Project Workflow
Dataset
   ↓
Load Dataset
   ↓
Explore Data
   ↓
Data Validation
   ↓
Train/Test Split
   ↓
TF-IDF Feature Extraction
   ↓
Train Machine Learning Models
   ↓
Evaluate Models
   ↓
Compare Models
   ↓
Select Best Model
   ↓
Build Prediction Function
   ↓
Predict Category + Confidence
Project Phases
Phase 1 – Dataset

The customer support ticket dataset is loaded using Pandas.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/content/customer_support_tickets_1540.csv")
Phase 2 – Load and Explore Data

The dataset is explored using:

df.head()
df.shape
df.columns
df.info()

The dataset contains:

1540 rows
4 columns

The columns are:

ticket_id
ticket
category
priority
Phase 3 – Data Validation

Missing values are checked:

df.isna().sum()

There are no missing values in the dataset.

Duplicate records are also checked:

df.duplicated().sum()

The dataset contains no duplicate records.

Phase 4 – Train/Test Split

The ticket text is used as the input feature and the category is used as the target.

X = df["ticket"]
y = df["category"]

The dataset is divided into training and testing data using an 80/20 split.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

The split produces:

Training data: 1232
Testing data: 308
Phase 5 – TF-IDF Feature Extraction

Machine Learning models cannot directly understand text.

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert customer ticket text into numerical features.

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

The resulting feature matrices are:

Training: (1232, 107)
Testing:  (308, 107)
Phase 6 – Logistic Regression

The first Machine Learning model is Logistic Regression.

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

The Logistic Regression model achieved approximately 100% accuracy on the test dataset.

Phase 7 – Random Forest Classifier

Random Forest is also trained using the TF-IDF features.

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train_tfidf, y_train)

rf_pred = rf_model.predict(X_test_tfidf)

print(
    "Random Forest Accuracy:",
    accuracy_score(y_test, rf_pred)
)

The Random Forest model achieved:

Random Forest Accuracy: 1.0

The classification report also shows 1.00 precision, recall, and F1-score for all seven categories on the test set.

Phase 8 – Model Comparison

The project compares:

Logistic Regression
Random Forest Classifier

Based on the project results, Random Forest is selected as the best model.

Best Model: Random Forest Classifier
Accuracy: 100%
Phase 9 – Ticket Classification

A prediction function is created to classify new customer support tickets.

def classify_ticket(ticket):
    ticket_tfidf = tfidf.transform([ticket])
    prediction = rf_model.predict(ticket_tfidf)
    return prediction[0]
Example 1

Input:

My credit card payment was declined

Output:

Predicted Category: Payment Issue
Example 2

Input:

I forgot my account password

Output:

Predicted Category: Account Issue




Phase 10 – Confidence Prediction

The project also calculates the confidence of the prediction.

def classify_ticket_(ticket):
    ticket_tfidf = tfidf.transform([ticket])

    prediction = rf_model.predict(ticket_tfidf)[0]

    probabilities = rf_model.predict_proba(ticket_tfidf)[0]

    confidence = max(probabilities)

    return prediction, confidence

Example:

Input:
I forgot my account password

Category:
Account Issue

Confidence:
0.99

Confidence:
99.0%




Model Evaluation

The models are evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

The confusion matrix shows the classification performance across the seven support categories.

Categories

The system can classify tickets into:

1. Payment Issue
2. Technical Issue
3. Order Issue
4. Subscription
5. Delivery
6. Refund
7. Account Issue
Example Predictions
Customer Ticket	Predicted Category
My credit card payment was declined	Payment Issue
I forgot my account password	Account Issue
My order has not arrived	Delivery
I want to cancel my subscription	Subscription
I need to return my order	Refund
