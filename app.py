from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words or word in ['like', 'love', 'enjoy']]
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        print(f"Original Message: {message}")  # Debugging line
        processed_message = preprocess_text(message)
        print(f"Processed Message: {processed_message}")  # Debugging line
        vect = vectorizer.transform([processed_message])
        prediction = model.predict(vect)
        print(f"Prediction: {prediction[0]}")  # Debugging line
        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
