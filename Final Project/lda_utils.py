# This is a container for all methods we need to use LDA models

from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt


def dictionary_LDA (corpus):
    """ This function creates a gensim dictionary out of a given corpus. The corpus needs to be tokenized. """
    return corpora.Dictionary(corpus)


def corpus(gensim_dictionary, corpus):
    return [gensim_dictionary.doc2bow(list_of_tokens) for list_of_tokens in corpus]


def LDA_model (corpus, num_topics, dictionary_LDA, passes =4, alpha=0.01, eta=0.01):
    return models.LdaModel(corpus, num_topics=num_topics,
                                id2word=dictionary_LDA,
                                passes=passes, alpha=[alpha]*num_topics,
                                eta=[eta]*len(dictionary_LDA.keys()))


def get_topics(lda_model, num_topics, num_words=10):
    # get the topics
    for i, topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=num_words):
        print(str(i) + ": " + topic)
    print()


def coherence_score(model, texts, dictionary_LDA, coherence="c_v"):
    coherence_model_lda = CoherenceModel(model=model,
                                         texts=texts,
                                         dictionary=dictionary_LDA,
                                         coherence=coherence)
    coherence_lda = coherence_model_lda.get_coherence()
    return coherence_lda


def plotting(x_limit, x_start, values, x_label, x_step=1):
    x = range(x_start, x_limit, x_step)
    plt.plot(x, values)
    plt.xlabel(x_label)
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.show()


def compare_num_topics(min_num_topics, max_num_topics, corpus, dictionary_LDA, texts, passes =4, alpha =0.01, eta=0.01, coherence="c_v"):

    coherence_values = []
    model_list = []
    for num_topics in range(min_num_topics, max_num_topics):
        model = LDA_model(corpus, num_topics, dictionary_LDA, passes, alpha, eta)
        model_list.append(model)
        coherence_values.append(coherence_score(model, texts, dictionary_LDA, coherence))
        print("Currently at:", num_topics, "number of topics")

    return coherence_values, model_list


def compare_alpha(min_alpha, max_alpha, steps, num_topics, corpus, dictionary_LDA, texts, passes =4, eta=0.01, coherence="c_v"):
    coherence_values=[]
    model_list = []
    alpha = min_alpha

    # while loop is needed as floats cannot be used in for loops
    while alpha < max_alpha:
        model = LDA_model(corpus, num_topics, dictionary_LDA, passes, alpha, eta)
        model_list.append(model)
        coherence_values.append(coherence_score(model, texts, dictionary_LDA, coherence))
        alpha += steps
        print("Currently at:", alpha, " Alpha")

    return coherence_values, model_list


def compare_eta(min_eta, max_eta, steps, num_topics, corpus, dictionary_LDA, texts, passes=4, alpha=0.01,
                coherence="c_v"):
    coherence_values = []
    model_list = []
    eta = min_eta

    # while loop is needed as floats cannot be used in for loops
    while eta < max_eta:
        model = LDA_model(corpus, num_topics, dictionary_LDA, passes, alpha, eta)
        model_list.append(model)
        coherence_values.append(coherence_score(model, texts, dictionary_LDA, coherence))
        alpha += steps
        print("Currently at:", alpha, " Alpha")

    return coherence_values, model_list
