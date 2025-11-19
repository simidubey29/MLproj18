import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# ensure nltk resources
nltk.download('stopwords')
nltk.download('punkt')

# load vectorizer, model, (optional) label encoder
tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
model = pickle.load(open('model1.pkl', 'rb'))
try:
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
except Exception:
    label_encoder = None

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("📱 SMS Spam Classifier")

input_sms = st.text_area("Enter message")

if st.button("Predict"):

    if not input_sms.strip():
        st.warning("Please enter a message.")
    else:
        # preprocessing
        transformed_sms = transform_text(input_sms)

        # vectorize
        vector_input = tfidf.transform([transformed_sms])

        # predict
        result = model.predict(vector_input)[0]

        # map prediction to readable label
        label = None
        if label_encoder is not None:
            try:
                # if model gives numeric label (0/1)
                label = label_encoder.inverse_transform([int(result)])[0]
            except Exception:
                # if model already returns string labels
                try:
                    label = label_encoder.inverse_transform([result])[0]
                except Exception:
                    label = str(result)
        else:
            # no encoder: handle common cases
            if isinstance(result, str):
                label = result
            else:
                try:
                    label = 'spam' if int(result) == 1 else 'ham'
                except Exception:
                    label = str(result)

        # final output
        if 'spam' in str(label).lower():
            st.header("🚨 This message is SPAM!")
        else:
            st.header("✅ This message is NOT SPAM")
