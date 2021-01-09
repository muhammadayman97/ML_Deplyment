from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import nltk
from nltk.downloader import download, download_shell 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle

# load the model from disk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.data.path.append('./nltk_data/')
filename = '/home/muhammad/Desktop/new_sent/nlp_model.pkl'
NB = pickle.load(open(filename, 'rb'))
tfidf_vectorizer = pickle.load(open('/home/muhammad/Desktop/new_sent/transform.pkl','rb'))
app = Flask(__name__, template_folder='template')
RUN python -m nltk.downloader popular


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = tfidf_vectorizer.transform(data).toarray()
		my_prediction = NB.predict(vect)
	return render_template('results.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
