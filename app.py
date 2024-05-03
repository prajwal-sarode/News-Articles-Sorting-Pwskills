from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def ValuePredictor(texts):
    m = pickle.load(open('model.pkl', 'rb'))
    with open('tfidf.pkl', 'rb') as file:
        tfidf = pickle.load(file)

    text_features = tfidf.transform(texts)
    predictions = m.predict(text_features)

    return predictions

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('a.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        texts = list(to_predict_list.values())
        results = ValuePredictor(texts)

        predictions = []
        for result in results:
            if result == 0:
                predictions.append('business')
            elif result == 1:
                predictions.append('entertainment')
            elif result == 2:
                predictions.append('politics')
            elif result == 3:
                predictions.append('sport')
            elif result == 4:
                predictions.append('tech')
            else:
                predictions.append('Unknown')

        return render_template('index.html', prediction=predictions)
    else:
        return "Please submit a form"

if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)
