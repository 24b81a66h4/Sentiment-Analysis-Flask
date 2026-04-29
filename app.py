from flask import Flask, render_template, request
import re
import nltk

app = Flask(__name__)

nltk.download('stopwords')
nltk.download('punkt')

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    cleaned_text = clean_text(text).lower()

    # simple logic (no sklearn at all)
    if any(word in cleaned_text for word in ['good','great','love','excellent','amazing']):
        sentiment = "Positive"
    elif any(word in cleaned_text for word in ['bad','worst','hate','poor','terrible']):
        sentiment = "Negative"
    else:
        sentiment = "ML Prediction System"

    return render_template('index.html', prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
