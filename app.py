
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

import streamlit as st
import pickle
from utils import get_spam_probability
import nltk
import os

NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

def download_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", download_dir=NLTK_DATA_DIR)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", download_dir=NLTK_DATA_DIR)

download_nltk()


# -----------------------------
# Load model & vectorizer
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("model1.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer1.pkl", "rb"))
    return model, vectorizer
model1, vectorizer1 = load_artifacts()

@st.cache_resource
def load_model():
    tfidf = pickle.load(open("vectorizer1.pkl", "rb"))
    model = pickle.load(open("model1.pkl", "rb"))
    return tfidf, model,

tfidf, model = load_model()



user_input = st.text_area("Enter text to analyze")
if st.button("Analyze"):
    prob = get_spam_probability(model1,vectorizer1, user_input)
    st.progress(int(prob*100))  # Convert to 0–100

    st.write(f"Spam Probability: {prob*100:.2f}%")

    percentage = round(prob * 100, 2)

    st.write(f"### Spam Probability: **{percentage}%**")
# ensure nltk_data dir is included and writable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
nltk_data_dir = os.path.join(BASE_DIR, "nltk_data")
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

def ensure_nltk_resource(resource_name, downloader_name=None):

    try:
        nltk.data.find(resource_name)
    except LookupError:
        to_download = downloader_name or resource_name.split('/')[-1]
        print(f"[NLTK] Resource {resource_name} not found. Downloading '{to_download}' to {nltk_data_dir} ...")
        nltk.download(to_download, download_dir=nltk_data_dir)


ensure_nltk_resource("tokenizers/punkt", downloader_name="punkt")
ensure_nltk_resource("tokenizers/punkt_tab/english", downloader_name="punkt_tab")
ensure_nltk_resource("corpora/stopwords", downloader_name="stopwords")



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
input_sms = st.text_area("Enter message", key="input_sms_1")


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

