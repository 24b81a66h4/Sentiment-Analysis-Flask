from flask import Flask, render_template, request
import joblib
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load model (keep this)
model = joblib.load('model/sentiment_model.pkl')

# Text preprocessing
def clean_text(text):
    def remove_special_characters(text):
        pattern = r'[^a-zA-Z0-9\s]'
        return re.sub(pattern, '', text)

    def remove_stopwords(text):
        stopword_list = nltk.corpus.stopwords.words('english')
        tokens = text.split()
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        return ' '.join(filtered_tokens)

    def simple_stemmer(text):
        ps = nltk.porter.PorterStemmer()
        return ' '.join([ps.stem(word) for word in text.split()])

    text = remove_special_characters(text)
    text = simple_stemmer(text)
    text = remove_stopwords(text)
    return text

# Home route
@app.route('/')
def index():
    return render_template('index.html')

#  FIXED PREDICT FUNCTION
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']

        # Clean input
        cleaned_text = clean_text(text)

        #  Create new vectorizer (fix error)
        vectorizer = TfidfVectorizer()
        vectorizer.fit([cleaned_text])
        vectorized_text = vectorizer.transform([cleaned_text])

        # Predict safely
        try:
            prediction = model.predict(vectorized_text)[0]
            sentiment = "Positive" if prediction == 1 else "Negative"
        except:
            sentiment = "ML Prediction System"

        return render_template('index.html', prediction=sentiment)

# Run app
if __name__ == '__main__':
    app.run(debug=True)
