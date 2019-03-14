import numpy as np
from utils import logzero


class HMM():
    def __init__(self, V, K):
        self._K = K
        self._V = V
        #count matrices
        self._init_counts = np.zeros((K), dtype=float)
        self._transition_counts = np.zeros((K,K), dtype=float)
        self._emission_counts = np.zeros((V,K), dtype=float)
        self._final_counts = np.zeros((K), dtype=float)
        #paramenters of the model probabillity matrices
        self._init_probs = np.zeros((K), dtype=float)
        self._transition_probs = np.zeros((K,K), dtype=float)
        self._emission_probs = np.zeros((V,K), dtype=float)
        self._final_probs = np.zeros((K), dtype=float)
        return


    def clear_counts(self, smoothing=0):
        """ Clear all the count tables."""
        self._init_counts.fill(smoothing)
        self._transition_counts.fill(smoothing)
        self._final_counts.fill(smoothing)
        self._emission_counts.fill(smoothing)
        return

    def compute_parameters(self):
        """ Estimate the HMM parameters by normalizing the counts."""

        # Normalize the initial counts.
        #TODO: compute self.initial_probs

        # Normalize transition counts
        #TODO: compute self.transition_probs

        # Normalize final counts
        #TODO: compute self.final_probs

        # Normalize emission counts
        #TODO: compute self.initial_probs

        return

    def compute_scores(self, sequence):
        length = sequence.len # Length of the sequence.
        num_states = self._K # Number of states of the HMM.

        # Initial position.
        initial_scores = np.log(self.initial_probs)

        # Intermediate position.
        emission_scores = np.zeros([length, num_states]) + logzero()
        for pos in range(length):
            emission_scores[pos, :] = np.log(self.emission_probs[sequence.x[pos], :])
        transition_scores = np.log(self.transition_probs)

        # Final position.
        final_scores = np.log(self.final_probs)

        return initial_scores, transition_scores, final_scores, emission_scores



    def train_supervised(self, dataset, smoothing=0):
        """ Train an HMM from a list of sequences containing observations
        and the gold states. This is just counting and normalizing.
        :param dataset: list of sequences each containing observations and labels"""
        # Set all counts to zeros (optionally, smooth).
        self.clear_counts(smoothing)
        # Count occurrences of events.
        self.collect_counts_from_corpus(dataset)
        # Normalize to get probabilities.
        self.compute_parameters()
        return

    def collect_counts_from_corpus(self, dataset):
        """
        Collects counts from a labeled corpus.
        :param dataset: list of sequences each containing observations and labels type Seq
        :return:
        """
        print("Training the HMM... total sequences: {}".format(len(dataset)))
        nreport = 500
        for i,sequence in enumerate(dataset):
            if i%nreport==0:
                print("seq {} out of {}.".format(i,len(dataset)))
            # Take care of first position.
            self._init_counts[sequence.y[0]] += 1
            self._emission_counts[sequence.x[0], sequence.y[0]] += 1

            # Take care of intermediate positions.
            for i, x in enumerate(sequence.x[1:]):
                y = sequence.y[i+1]
                y_prev = sequence.y[i]
                self._emission_counts[x, y] += 1
                self._transition_counts[y, y_prev] += 1

            # Take care of last position.
            self._final_counts[sequence.y[-1]] += 1
        return