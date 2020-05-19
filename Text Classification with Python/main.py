import pandas as pd
import pymysql
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import pickle


							# transforms Xtext into an array that counts keywords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import sent_tokenize
from sklearn.externals import joblib



def save_result(result, name, email, phone):

	for i in range(0, len(result)):
		sql = "INSERT INTO `results`(`name`, `email`,`phone`,`word`, `value`) VALUES ('" + name + "','" + email + "','" + phone + "', '" + \
			  result[i].split(' - ')[0] + "', '" + result[i].split(' - ')[1] + "')"
		conn = pymysql.connect(host='localhost', user='root', password='', db='travel')
		link = conn.cursor()
		result2 = link.execute(sql)
		conn.commit()

		#can remove classify_word


# and weighs frequently used words accordingly.
def predict(text, name, email, phone):   #can remove name email phone


	yelp = pd.read_csv('yelp.csv')
	yelp['text_length'] = yelp['text'].apply(len)							# Add text length as a column in dataframe

	### NLP Classification

	yelp_class = yelp				# create new dataframe of just 1 and 5 star reviews.

	Xtext, ystars = yelp_class['text'], yelp_class['stars']					# Define features as Xtext, and target as ystars.

	cv = CountVectorizer()													# Create a count vectorizer object
	Xtext = cv.fit_transform(Xtext)
	### Model Training
	try:
		nb = joblib.load('filename.pkl')
	except Exception:
		Xtext_train, Xtext_test, ystars_train, ystars_test =\
			train_test_split(Xtext, ystars, test_size=0.30)

		nb = MultinomialNB()
		nb.fit(Xtext_train, ystars_train)
		joblib.dump(nb, 'filename.pkl')

	#nbPredict = nb.predict(Xtext_test)
	#
	# print(confusion_matrix(ystars_test, nbPredict))
	# print('\n\n')
	#print(classification_report(ystars_test, nbPredict))

	# text = "The park was a very beautiful place. The food was really good. But the weather was awful. Transportation was dirty."

	tokenized_text = sent_tokenize(text)
	result = []
	for sentence in tokenized_text:
		print(sentence)
		textt = word_tokenize(sentence)
		words = ""
		tagged = pos_tag(textt)
		for word,pos in tagged:
			if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
				words += word + ' '
		result.append(str(words) + " - " + str(nb.predict(cv.transform([sentence]))[0]))

	#can be removed
	save_result(result, name, email, phone)

	return result
# print(nb.predict())