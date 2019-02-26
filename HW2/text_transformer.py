## This is a container of methods to simplify the transformation of text file to do proper mining on texts
#  necessrary imports
import nltk
import string
import re
nltk.download("punkt")
#nltk.download("wordnet")
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

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
def normalizer(token_list):
    punct = "\{}".format("|\\".join(string.punctuation))
    output_list = [token.lower() for token in token_list 
                                if token not in string.punctuation and 
                                not re.search( r'{} +'.format(punct), token) and
                                token != " " and
                                token not in stopwords.words("english")]
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
""" def count_frequencies(token_list):
    if any(isinstance(el, str) for el in token_list):
        return [nltk.FreqDist(token_list)[token] for token in nltk.FreqDist(token_list)]
    elif any(isinstance(el, list) for el in token_list):
        flatten = [element for token in token_list for element in token] """

## Flattener
#  for all preprocessing tasks we want lists of strings
#  this function transforms list in lists into this type
""" def flatten (word_list):
    if any(isinstance(el, list) for el in word_list):
        flatten_list = [element for sub_list in word_list for element in sub_list]
        return flatten_list
    elif any(isinstance(el, str) for el in word_list):
        return word_list
    else:
        print("Something's wrong") """