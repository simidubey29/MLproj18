import string
import nltk


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# ---------------------------------------
# 1. Clean and preprocess the text
# ---------------------------------------
def transform_text(text):
    text = text.lower()

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove punctuation & stopwords, apply stemming
    cleaned = []
    for word in tokens:
        if word.isalnum():  # remove punctuation
            if word not in stopwords.words("english"):
                cleaned.append(ps.stem(word))

    return " ".join(cleaned)


# ---------------------------------------
# 2. Get Spam Probability
# ---------------------------------------
def get_spam_probability(model, vectorizer, text):
    """
    Returns probability that the input text is spam.
    Model must be a trained classifier with predict_proba().
    """

    if not text or text.strip() == "":
        return 0.0    # Empty text → 0% spam

    # Preprocess
    transformed_text = transform_text(text)

    # Vectorize
    vector_input = vectorizer.transform([transformed_text])

    # Predict probability
    proba = model.predict_proba(vector_input)[0][1]  # class index 1 = spam

    return float(proba)
