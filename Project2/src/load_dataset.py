import numpy as np
import csv
import re,string
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.classify.textcat import TextCat
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#need to install nltk packages punkt and corpus.stopwords
# nltk.download('wordnet')
## nltk.download('crubadan')

def load_data(path ='../data/reddit_train.csv', preprocess= True,  stem=True,lemmatize=True, translate = False, bSentiment = False):
    """
    Loads the training set and preprocesses it

    Parameters
    --------
    path: string
        path to csv dataset

    preprocess: boolean
        whether to preprocess or not
    
    shuffle: boolean
        whether shuffle the array or not
    
    Returns
    ----------
        headers:np array
            headers (id,comments,subreddits)
        data:np.array
            the data array corresponding to the headers
    """

    with open(path, encoding='utf8', mode='r') as f:
        data = list(csv.reader(f, delimiter=','))
    
    headers = data[0]
    
    data = np.asarray(data[1:], dtype=np.str)
    
    if bSentiment:
        sentiment = _sentiment_(data)
    
    if preprocess:
        data[:,1] = _preprocess_(data,stem=stem,lemmatize=lemmatize,translate=translate)[:]
        
    if bSentiment:
        return headers, data, sentiment
    
    return headers, data,None

def _sentiment_(data):
    comments = np.copy(data[:,1])
    sia = SentimentIntensityAnalyzer()
    sentiment = []
    for i in comments:
        sentiment.append(sia.polarity_scores((i))['compound'])
   
    return sentiment
               
                
def _preprocess_(dataset,stem=True,lemmatize=True,translate=True):
    comments = np.copy(dataset[:,1])
        
    #merge both sets of stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    stop_words.update(set(ENGLISH_STOP_WORDS))

    english_vocab = set(w.lower() for w in nltk.corpus.words.words())

    lemmatizer=WordNetLemmatizer()
    stemmer = PorterStemmer()
    #tc = TextCat()

    good_comments = []

    #TODO add handling of links. ie 'https://'
    for comment in comments:
        url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', comment) #find links?
        if url!=[]:
            for u in url:
                comment = comment.replace(u,' https ') 
                
        comment = comment.lower()
        comment = re.sub(r'\d','',comment) #reduce any digits
        comment = comment.translate(str.maketrans("","", string.punctuation)) #remove punctuation
        comment = re.sub(r'\s+',' ', comment).strip() #remove any double (or more) spaces

        #comment_lang = tc.guess_language(comment)
        #print(comment_lang)
        #if translate and comment_lang!='eng':
        #    try:
        #        print('found language:',comment_lang)
        #        comment = str(TextBlob(comment).translate(from_lang=comment_lang,to='en'))
        #    except Exception as e:
        #        print('exception in translation',e)
        #        print(comment)
        #        pass
        tokens = word_tokenize(comment) #tokenize words


        if stem and lemmatize:
            result = [stemmer.stem(lemmatizer.lemmatize(i)) for i in tokens if not i in stop_words] #lemmatize and stem words if not a stop word
        elif stem and not lemmatize:
            result = [stemmer.stem(i) for i in tokens if not i in stop_words]
        elif not stem and lemmatize: 
            result = [lemmatizer.lemmatize(i) for i in tokens if not i in stop_words]
        else: 
            result = [i for i in tokens if not i in stop_words] #only remove stop words
        
        final_comment = ' '.join(result)
        good_comments.append(final_comment)
    return good_comments
    

    
if __name__ == '__main__':
    #_,data = load_train(preprocess=False)
    headers,data = load_data(translate=True)
    print(data)
