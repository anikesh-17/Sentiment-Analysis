import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string
import pickle

# Download stopwords
nltk.download('stopwords')

# Load dataset
data = pd.read_csv('sentiment_data.csv')

# Convert labels: 0 -> 'negative', 1 -> 'positive'
label_mapping = {0: 'negative', 1: 'positive'}
data['label'] = data['label'].map(label_mapping)

# Preprocess the data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words or word in ['like', 'love', 'enjoy']]
    return ' '.join(words)

data['processed_text'] = data['text'].apply(preprocess_text)

# Split the data
X = data['processed_text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=5)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Build and train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vectorized, y_train)

# Test the model
y_pred = model.predict(X_test_vectorized)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
pickle.dump(model, open('sentiment_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

