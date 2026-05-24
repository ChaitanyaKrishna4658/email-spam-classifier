# Email Spam Classifier 📧

## Installing Required Libraries

```python
!pip install pandas
!pip install scikit-learn
!pip install nltk
```

---

## Upload CSV File to Google Colab

```python
from google.colab import files

uploaded = files.upload()
```

### Output
```text
spam.csv
Saving spam.csv to spam.csv
```

---

# Python Code

```python
# Import Libraries

import pandas as pd  
# Used for handling datasets

from sklearn.feature_extraction.text import CountVectorizer  
# Converts text into numerical format

from sklearn.model_selection import train_test_split  
# Used for splitting dataset into training and testing data

from sklearn.naive_bayes import MultinomialNB  
# Naive Bayes algorithm used for classification

from sklearn.metrics import accuracy_score  
# Used for checking model accuracy


# Load Dataset

data = pd.read_csv("spam.csv", encoding='latin-1')
# Reads CSV dataset
# latin-1 helps handle special characters


# Keep Required Columns

data = data[['v1', 'v2']]
# v1 = label (spam/ham)
# v2 = message text


# Rename Columns

data.columns = ['label', 'message']
# Renaming columns for better understanding


# Convert Labels into Numbers

data['label'] = data['label'].map({
    'ham': 0,
    'spam': 1
})
# ham = 0
# spam = 1
# This process is called Label Encoding


# Input and Output

x = data['message']
# Input messages

y = data['label']
# Output labels


# Convert Text into Vectors

cv = CountVectorizer()
# Converts words into numerical vectors

x = cv.fit_transform(x)


# Split Dataset

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# 80% data used for training
# 20% data used for testing


# Create Model

model = MultinomialNB(alpha=0.1)
# Naive Bayes classifier model


# Train Model

model.fit(x_train, y_train)
# Model learns from training data


# Prediction

y_pred = model.predict(x_test)
# Predicts test data


# Accuracy Checking

print("===== EMAIL SPAM CLASSIFIER =====")

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")


# Full Email Testing

email_text = """
Subject: College Project Submission

Hello Chaitanya,

Your AIML project submission has been successfully received.

Please attend tomorrow's review meeting at 11:00 AM in Lab 3.

Bring your project documentation and presentation slides.

Thank you.

Regards,
Project Coordinator
"""


# Convert Email into Vector

email_vector = cv.transform([email_text])


# Predict Email

prediction = model.predict(email_vector)


# Final Output

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam Message")
```

---

# Output

```text
===== EMAIL SPAM CLASSIFIER =====

Model Accuracy: 98.03 %

Not Spam Message
```

---

# Spam Email Demo

```text
Subject: Congratulations Winner!

Dear Customer,

You have been selected as the lucky winner of a free iPhone 16.

Click the link below to claim your reward immediately.

Limited offer. Hurry up!

Regards,
Prize Team
```

Expected Output:

```text
Spam Message
```

---

# Machine Learning Workflow

```text
Dataset
   ↓
Text Preprocessing
   ↓
Convert Text into Numbers
   ↓
Train Model
   ↓
Spam / Not Spam Prediction
```
