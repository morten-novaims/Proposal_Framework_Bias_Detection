import numpy as np
from IPython import embed
from utils import *
from HMM import HMM

def read_conll(filename, word_vocab=None, label_vocab=None):
    """
    reads conll file and outputs list of sentences and labels
    :param filename: string filepath
    :param word_vocab: word dictionary
    :param label_vocab: label dictionary
    :return: list of words and labels as a seq class and the word and label dictionary
    """
    fin = open(filename, 'r')
    sentences = []
    if word_vocab is None:
        word_vocab = stringdict()
    if label_vocab is None:
        label_vocab = stringdict()

    words = []
    labels = []

    for line in fin.readlines():
        line = line.rstrip()
        tokens = line.split()
        if len(tokens) > 0:
            word = tokens[0]
            label = tokens[3]
            if word == "-DOCSTART-":
                continue
            words.append(word_vocab.add(word))
            labels.append(label_vocab.add(label))
        else:
            #end of sentence
            if len(words)>0:
                sentences.append(Seq(words, labels))
            words = []
            labels = []
    return sentences, word_vocab, label_vocab


class Viterbi():
    def decode_sequence(self, initial_scores, transition_scores, final_scores, emission_scores):
        length = np.size(emission_scores, 0)  # Length of the sequence.
        num_states = np.size(initial_scores)  # Number of states.

        # Variables storing the Viterbi scores.
        viterbi_scores = np.zeros([length, num_states]) + logzero()

        # Variables storing the paths to backtrack.
        viterbi_paths = -np.ones([length, num_states], dtype=int)

        # Most likely sequence.
        best_path = -np.ones(length, dtype=int)

        #  Initialize the viterbi scores: viterbi(1, c_k )
        viterbi_scores[0, :] = emission_scores[0, :] + initial_scores

        #TODO: implement the algorithm here!!

        for i in range(1, length):              # loop over all sentences
            for state in range(num_states):         # loop over all states/labels
                # get the max value of all transition scores of states
                p_trans = transition_scores[state, :] + viterbi_scores[i-1, :]
                viterbi_scores[i, state] = np.max(p_trans) + emission_scores[i, state]
                # get the backtrack path
                viterbi_paths[i, state] = np.argmax(p_trans)

        # Termination: viterbi(N + 1, stop)
        list_final_scores = viterbi_scores[length - 1, :] + final_scores
        best_score = np.max(list_final_scores)
        best_path = viterbi_paths[:, (np.argmax(list_final_scores))]
        return best_path, best_score


class SequenceClassifier():
    def __init__(self, model, decoder):
        self._model = model
        self._decoder = decoder
        return

    def decode(self, sequence):
        """Compute the most likely sequence of states given the observations,
        by running the Viterbi algorithm."""

        # Compute scores given the observation sequence.
        # basically apply np.log to the probs - more or less
        initial_scores, transition_scores, final_scores, emission_scores = \
            self._model.compute_scores(sequence)

        # Run the forward algorithm.
        best_states, total_score = self._decoder.decode_sequence(initial_scores,
                                                            transition_scores,
                                                            final_scores,
                                                            emission_scores)

        predicted_sequence = sequence.copy()
        predicted_sequence._labels = best_states # a better way would be to code it through properties, this also works
        return predicted_sequence, total_score

    def decode_corpus(self, dataset):
        """Run viterbi_decode at corpus level."""

        print("decoding corpus")
        predictions = []
        for i,sequence in enumerate(dataset):
            if i%100==0:
                print("decoding sequence {} out of {}".format(i, len(dataset)))
            predicted_sequence, _ = self.decode(sequence)
            predictions.append(predicted_sequence)
        return predictions

    def evaluate_corpus(self, dataset, predictions):
        """Evaluate classification accuracy at corpus level, comparing with
        gold standard."""
        print("evaluating corpus")
        total = 0.0
        correct = 0.0
        for i, sequence in enumerate(dataset):
            pred = predictions[i]
            for j, y_hat in enumerate(pred.y):
                if sequence.y[j] == y_hat:
                    correct += 1
                total += 1
        return correct / total


if __name__ == "__main__":
    datasetpath = "CONLL2003/"
    train_data, dV, dL = read_conll(datasetpath + "train.txt")
    dV.freeze()
    dL.freeze()
    test_data,_,_ = read_conll(datasetpath + "test.txt", word_vocab=dV, label_vocab=dL)


    #train the model with the dataset supervisedly
    hmm = HMM(dV.len, dL.len)
    hmm.train_supervised(train_data, smoothing=1e-10)
    print("done training")
    #now you can check what are the probabilities, init, transition, emission, and final
#    embed()
    seq_classifier = SequenceClassifier(model=hmm, decoder=Viterbi())

    train_predictions = seq_classifier.decode_corpus(train_data)
    train_acc = seq_classifier.evaluate_corpus(train_data, train_predictions)
    print("training accuracy: {}".format(train_acc))

    test_predictions = seq_classifier.decode_corpus(test_data)
    test_acc = seq_classifier.evaluate_corpus(test_data, test_predictions)
    print("testing accuracy: {}".format(test_acc))


#    embed()