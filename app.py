from flask import Flask, render_template, request, jsonify
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np

app = Flask(__name__)

model = pickle.load(open('classifier.pkl', 'rb'))
w2v_model = Word2Vec.load('word2vec.model')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Function to clean and preprocess the input text
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = word_tokenize(review)
    review = [word for word in review if word not in stopwords.words('english')]
    return review

def predict(text):
    review = preprocess_text(text)
    
    # Use Word2Vec for feature extraction
    review_vect = np.array([w2v_model.wv[word] for word in review if word in w2v_model.wv.key_to_index])
    
    # Check if the array is empty
    if review_vect.size == 0:
        return 'REAL'
    
    review_vect = np.mean(review_vect, axis=0)
    
    # Update the number of features used for prediction
    if review_vect.shape[0] != 100:
        review_vect = review_vect[:100]  # Keep only the first 100 features if there are more

    # Reshape the vector to match the shape used during training
    review_vect = review_vect.reshape(1, -1)

    prediction = 'FAKE' if model.predict(review_vect) == 0 else 'REAL'
    return prediction


@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)

@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)

if __name__ == "__main__":
    app.run()
