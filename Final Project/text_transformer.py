## This is a container of methods to simplify the transformation of text file to do proper mining on texts
#  necessrary imports
import nltk
import string
import re
import langdetect
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import os
import pickle

## Tokenizer
#  This tokenizer returns the text split into every single word
def tokenizer(text):
    return nltk.word_tokenize(text)

## Normalizer
#  This normalizer removes all punctuation (even consecutive punctuation) and whitespaces
#  translates everything into lowercase
#  remove all stopwords based on this library:
#  digits stay as they are
#  the method takes a tokenized list as an input
def normalizer(token_list, add_words=[]):
    punct = "\{}".format("|\\".join(string.punctuation))
    output_list = [token.lower() for token in token_list 
                                if token not in string.punctuation and 
                                not re.search( r'{} +'.format(punct), token) and
                                token != " " and
                                token.lower() not in stopwords.words("english") and
				                token.lower() not in add_words]
    return output_list

## Lemmatizer
#  remove any inflectional endings or variant forms
#  example: am, are, is -> be
def lemmatizer(token_list):
    return [WordNetLemmatizer().lemmatize(token) for token in token_list]

## Stemmer
#  removes all affixes from stem
#  example: revival -> revive; operator -> operate
def stemmer(token_list):
    return [PorterStemmer().stem(token) for token in token_list]

## Undigitizer
#  change all digits to the word digit
#  example: 6 -> digit
def undigitizer(token_list):
    return [re.sub(r'\d+.', "digit", token) for token in token_list]

## Count Frequencies
#  counts the frequencies of unique words and returns a sorted list
#  this is neccessary for the bag of words approach
def count_frequencies(token_list):
    return [(token, token_list.count(token)) for token in set(token_list) ]

## Flattener
#  for all preprocessing tasks we want lists of strings
#  this function transforms list in lists into this type
def flatten(word_list):
    for word in word_list:
        if type(word) is list:
            yield from flatten(word)
        else:
            yield word

## Check words
#  check if words are in a corpus
#  used for filtering news articles
def check_words(word_list, corpus, method):
#""" This function checks for a string list of words if they are in the corpus. The corpus needs to be an untokenized single string. The default method is any. If all words need to be in the corpus the argument "all" needs to be passed."""
    if method(word in corpus for word in word_list):
        return True
    else:
        return False

# check if the len of the article is below 300
def check_len(article_text):
    if len(article_text) < 300:
        return True
    else:
        return False
    
# check if the article is of a particular language
def check_lang(article_text, lang):
    if langdetect.detect(article_text) != lang:
        return True
    else:
        return False

# check if the url is of an article or other thing like comments section, video, discussion, etc...
def check_not_news_url(article_url):
    if any(word in article_url for word in ["#comment","#video"]):
        return True
    else:
        return False

## Preprocessing Pipeline
def preprocessing(directory, verbose=False, remove_words=[], filter_words =[], 
                  filter_method=any, stemming=False, lemmatizing=False, lang='en'):
    """ Returns the preprocessing pipeline utilizing the text_transformer package.
    Additionally words can be passed, that should be removed. Furthermore, words can be passed, which have to be in the article (Filter).
    Returns corpus and dictionary. """

    article_list = os.listdir(directory)

    articles = []
    corpus = []
    count_len = 0
    count_filter = 0

    for i, article_name in enumerate(article_list):
        article = pickle.load( open("articles/"+article_name, "rb" ) )
        
        if i%1000 == 0:
            if verbose:
                print("We're at "+ str(round(i/len(article_list)*100,2))+ "% of the data.")

        # remove all articles shorter than 300 characters
        if check_len(article["text"]):
            count_len += 1 
            continue
           
        # remove all articles not of lang
        if check_not_news_url(article["url"]):
            count_len += 1 
            continue    
            
        # apply the filtering of words
        if len(filter_words) > 0:   # check if argument was passed
            if check_words(filter_words, article["text"], filter_method) == False:
                count_filter += 1
                continue
        
        # remove all articles not of lang
        if check_lang(article["text"], lang):
            count_len += 1 
            continue    

        # tokenize the text
        token_list = tokenizer(article["text"])

        # normalize the text
        token_list = normalizer(token_list, remove_words)

        # apply stemming if needed
        if stemming:
            token_list = stemmer(token_list)

        # apply lemmatizing if needed
        if lemmatizing:
            token_list = lemmatizer(token_list)

        corpus.append(token_list)
        articles.append(article)
    
    if verbose:
        print("Articles used: "+ str(round((len(articles) / len(article_list))*100, 2))+ " %")
        print("Articles used: "+ str(len(articles))+ "/"+str(len(article_list)))
        print("*" *45)
        print(count_len, " (", round(count_len/len(article_list)*100, 2), "%) Articles were filtered out because of length and language")
        print(count_filter," (", round(count_filter/len(article_list)*100, 2), "%) Articles were filtered out because of the filter words.")
        
    return articles, corpus