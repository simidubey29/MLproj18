def get_spam_probability(model, vect_msg):
    try:
        prob = model.predict_proba(vect_msg)[0][1]  # spam probability
        return prob
    except:
        return 0.0