import streamlit as st
import pickle
import re
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

nltk.download('punkt')
nltk.download('stopwords')

# loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)


    cleaned_tokens = []
    for token in doc:
        # Remove URLs
        if token.like_url:
            continue
        # Remove hashtags and mentions
        if token.text.startswith('#') or token.text.startswith('@'):
            continue
        # Remove special characters and punctuation
        if not token.is_alpha:
            continue
        cleaned_tokens.append(token.text)

    # Join the cleaned tokens back into a text
    cleaned_text = ' '.join(cleaned_tokens)

    return cleaned_text

# web app
def main():
    st.title('Resume Screening App')
    upload_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        clean_resume = preprocess(resume_text)

        # Transform the cleaned resume using the trained TfidfVectorizer
        input_features = tfidf.transform([clean_resume])

        # Make the prediction using the loaded classifier
        prediction_id = clf.predict(input_features)[0]
        st.write(prediction_id)


# python main
if __name__ == "__main__":
    main()
