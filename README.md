!pip install pandas
!pip install scikit-learn
!pip install nltk

Requirement already satisfied: pandas in /usr/local/lib/python3.12/dist-packages (2.2.2)
Requirement already satisfied: numpy>=1.26.0 in /usr/local/lib/python3.12/dist-packages (from pandas) (2.0.2)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas) (2026.1)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (1.6.1)
Requirement already satisfied: numpy>=1.19.5 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (2.0.2)
Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (1.16.3)
Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (1.5.3)
Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (3.6.0)
Requirement already satisfied: nltk in /usr/local/lib/python3.12/dist-packages (3.9.1)
Requirement already satisfied: click in /usr/local/lib/python3.12/dist-packages (from nltk) (8.3.3)
Requirement already satisfied: joblib in /usr/local/lib/python3.12/dist-packages (from nltk) (1.5.3)
Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.12/dist-packages (from nltk) (2025.11.3)
Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (from nltk) (4.67.3)

# CSV file to colab 
from google.colab import files

uploaded = files.upload()

#Output
spam.csv
spam.csv(text/csv) - 503663 bytes, last modified: 5/24/2026 - 100% done
Saving spam.csv to spam.csv

# Python Code
import pandas as pd  # Used for handling datasets
from sklearn.feature_extraction.text import CountVectorizer #Converts text into numbers
from sklearn.model_selection import train_test_split # Splitting of datasets
from sklearn.naive_bayes import MultinomialNB #used for classification
from sklearn.metrics import accuracy_score # for accuracy checking

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1') # reads csv datset and latin helps to special char

# Keep required columns
data = data[['v1', 'v2']] # for spam and non spam (2 cols)

# Rename columns
data.columns = ['label', 'message'] # renaming of variables 

# Convert labels into numbers
data['label'] = data['label'].map({
    'ham': 0,
    'spam': 1  # labels to numbers (Lable encoding)
})

# Input and output
x = data['message']
y = data['label']

# Convert text into vectors
cv = CountVectorizer() # words into numbers 
x = cv.fit_transform(x)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split( # x_train training messages, x_text training messages
    x,
    y,
    test_size=0.2, # 20% data for testing 80% for training
    random_state=42
)

# Create model
model = MultinomialNB(alpha=0.1)

# Train model 
model.fit(x_train, y_train) #learns by training messages

# Prediction
y_pred = model.predict(x_test) # And predits the test data 
print("===== EMAIL SPAM CLASSIFIER =====")
# Accuracy
accuracy = accuracy_score(y_test, y_pred) 

print("Model Accuracy:", round(accuracy * 100, 2), "%")

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

email_vector = cv.transform([email_text])

# Predict
prediction = model.predict(email_vector)


if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam Message")

Output Cell:
===== EMAIL SPAM CLASSIFIER =====
Model Accuracy: 98.03 %
Not Spam Message


for Spam Message demo:

# Subject: Congratulations Winner!

# Dear Customer,

# You have been selected as the lucky winner of a free iPhone 16.

# Click the link below to claim your reward immediately.

# Limited offer. Hurry up!

# Regards,
# Prize Team