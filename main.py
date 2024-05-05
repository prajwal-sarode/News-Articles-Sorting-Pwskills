import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def ValuePredictor(texts):
    m = pickle.load(open('model.pkl', 'rb'))
    with open('tfidf.pkl', 'rb') as file:
        tfidf = pickle.load(file)

    text_features = tfidf.transform(texts)
    predictions = m.predict(text_features)

    return predictions

def main():
    st.title('News Headline Prediction')

    news_headline = st.text_input('Enter News Headline')

    if st.button('Predict'):
        if news_headline:
            result = ValuePredictor([news_headline])
            prediction = map_prediction(result[0])
            st.write(f'The news belongs to class: {prediction}')
        else:
            st.write('Please enter a news headline')

def map_prediction(prediction):
    classes = ['business', 'entertainment', 'politics', 'sport', 'tech', 'Unknown']
    if prediction >= 0 and prediction < len(classes):
        return classes[prediction]
    else:
        return 'Unknown'

if __name__ == '__main__':
    main()
