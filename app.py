import flask
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# Use pickle to load in the pre-trained model
with open(f'model/tfidf.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

with open(f'model/sentiment_classifier.pkl', 'rb') as f:
    model = pickle.load(f)    

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        user_set = flask.request.form['user_input']
        # print(user_set)
        test_set = [user_set]
        # print(test_set)
        new_test = vectorizer.transform(test_set)
        prediction = model.predict(new_test)
        # print(prediction)
        return flask.render_template('main.html',
        							   result=prediction,
                                     )

if __name__ == '__main__':
    app.run()