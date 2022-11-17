from nltk.corpus import stopwords
import nltk 
import re
import string

nltk.download('stopwords')

stemmer = nltk.SnowballStemmer('english')
stopword = set(stopwords.words('english'))

def clean_text(text):
    """
    Params: text that will be cleaned
    Return: cleaned text
    """
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text