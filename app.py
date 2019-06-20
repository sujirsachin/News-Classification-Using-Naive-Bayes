from flask import Flask,render_template,url_for,request
from sklearn.model_selection import train_test_split

global my_prediction
global data

app = Flask(__name__)


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():



    import pandas as pd
    import numpy as np
    from collections import defaultdict
    import re

    from nltk.stem import PorterStemmer


    cachedStopWords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

    ps = PorterStemmer()
    def preprocess_string(string):

        cleaned_str=re.sub('[^a-z\s]+',' ',string,flags=re.IGNORECASE)
        cleaned_str=re.sub('(\s+)',' ',cleaned_str)
        cleaned_str=cleaned_str.lower()

        cleaned=ps.stem(cleaned_str)

        cleaned_str1= ' '.join([word for word in cleaned.split() if word not in cachedStopWords])


        return cleaned_str1


    class MultinomialNaiveBayes:

        def __init__(self,unique_classes):

            self.classes=unique_classes


        def bagOfWords(self,headline,dict_index):


            if isinstance(headline,np.ndarray): headline=headline[0]

            for token_word in headline.split(): #each word in preprocessed string

             self.bow_dicts[dict_index][token_word]+=1 #increment in its count

        def train(self,dataset,labels):

         self.headline=dataset
         self.category=labels
         self.bow_dicts=np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])


         if not isinstance(self.headline,np.ndarray): self.headline=np.array(self.headline)
         if not isinstance(self.category,np.ndarray): self.category=np.array(self.category)

        #Developing Bag of Words for each category
         for cat_index,cat in enumerate(self.classes):

            all_cat_headline=self.headline[self.category==cat] #filter all headline of category == cat

            #removing stopwords,tokenizing headlines

            cleaned_headline=[preprocess_string(cat_headline) for cat_headline in all_cat_headline]

            cleaned_headline=pd.DataFrame(data=cleaned_headline)

            #Bag of words for particular category
            np.apply_along_axis(self.bagOfWords,1,cleaned_headline,cat_index)



            prob_classes=np.empty(self.classes.shape[0])
            all_words=[]
            cat_word_counts=np.empty(self.classes.shape[0])
            for cat_index,cat in enumerate(self.classes):

                #Calculating prior probability p(c) for each class
                prob_classes[cat_index]=np.sum(self.category==cat)/float(self.category.shape[0])

                #Calculating total counts of all the words of each class

                cat_word_counts[cat_index]=np.sum(np.array(list(self.bow_dicts[cat_index].values())))+1 # |v| is remaining to be added

                #get all words of this category
                all_words+=self.bow_dicts[cat_index].keys()


            #combine all words of every category & make them unique to get vocabulary -V- of entire training set

            self.vocab=np.unique(np.array(all_words))
            self.vocab_length=self.vocab.shape[0]

            #computing denominator value
            denoms=np.array([cat_word_counts[cat_index]+self.vocab_length+1 for cat_index,cat in enumerate(self.classes)])


            self.cats_info=[(self.bow_dicts[cat_index],prob_classes[cat_index],denoms[cat_index]) for cat_index,cat in enumerate(self.classes)]
            self.cats_info=np.array(self.cats_info)


        def getHeadlineProb(self,test_headline):


            likelihood_prob=np.zeros(self.classes.shape[0]) #to store probability w.r.t each class

            #finding probability w.r.t each class of the given test example
            for cat_index,cat in enumerate(self.classes):

                for test_token in test_headline.split(): #split the test example and get p of each test word


                    #get total count of this test token from it's respective training dict to get numerator value
                    test_token_counts=self.cats_info[cat_index][0].get(test_token,0)+1


                    test_token_prob=test_token_counts/float(self.cats_info[cat_index][2])

                #To prevent underflow, log the value
                    likelihood_prob[cat_index]+=np.log(test_token_prob)

            post_prob=np.empty(self.classes.shape[0])
            for cat_index,cat in enumerate(self.classes):
                post_prob[cat_index]=likelihood_prob[cat_index]+np.log(self.cats_info[cat_index][1])

            return post_prob


        def test(self,test_set):

            predictions=[] #to store prediction of each headline
            for headline in test_set:


                cleaned_headline=preprocess_string(headline)

                #get the posterior probability of every headline
                post_prob=self.getHeadlineProb(cleaned_headline) #get prob of this headline for both classes


                predictions.append(self.classes[np.argmax(post_prob)])

            return np.array(predictions)

    training_set=pd.read_csv('news.csv',sep=',') # reading the training data-set

    #getting training set headline labels
    y_train=training_set['Category'].values
    x_train=training_set['Title'].values
    train_data,test_data,train_labels,test_labels=train_test_split(x_train,y_train,shuffle=True,test_size=0.25,random_state=42,stratify=y_train)
    classes1=np.unique(train_labels)
    nb=MultinomialNaiveBayes(classes1)
    nb.train(train_data,train_labels)
    pclasses=nb.test(test_data)
    acc1=np.sum(pclasses==test_labels)/float(test_labels.shape[0])

    acc=round(acc1,2)
    classes=np.unique(y_train)


    # Training phase....

    nb=MultinomialNaiveBayes(classes)
    nb.train(x_train,y_train)




    if request.method == 'POST':
	    message = request.form['message']
	    data = [message]

	    my_prediction=nb.test(data)



    if(my_prediction[0]==1):
        return render_template('sports.html',prediction = my_prediction,data=data[0], acc=acc)
    if(my_prediction[0]==2):
        return render_template('politics.html',prediction = my_prediction,data=data[0], acc=acc)
    if(my_prediction[0]==3):
        return render_template('tv.html',prediction = my_prediction,data=data[0], acc=acc)
    if(my_prediction[0]==4):
        return render_template('intnews.html',prediction = my_prediction,data=data[0], acc=acc)
    if(my_prediction[0]==5):
        return render_template('tech.html',prediction = my_prediction,data=data[0], acc=acc)
    if(my_prediction[0]==6):
        return render_template('business.html',prediction = my_prediction,data=data[0], acc=acc)
    if(my_prediction[0]==7):
        return render_template('health.html',prediction = my_prediction,data=data[0], acc=acc)


if __name__ == '__main__':
	app.run(debug=True)
