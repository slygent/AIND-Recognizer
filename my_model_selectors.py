import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str, n_constant=3, min_n_components=2, max_n_components=10, random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        component_range = range(self.min_n_components, self.max_n_components + 1)

        model_scores = [None] * len(component_range)

        for number_of_components in component_range:
            try:
                model_score = -(2 * self.base_model(number_of_components).score(self.X, self.lengths)) + (
                        number_of_components**2
                        + ((2 * number_of_components * self.X.shape[1]) - 1)
                        * np.log2(self.X.shape[0])
                )
            except (ValueError, AttributeError) as e:
                continue
            model_scores[number_of_components - self.min_n_components] = model_score

        if all(x is None for x in model_scores): return None

        best_model = model_scores.index(min(filter(None, model_scores))) + self.min_n_components

        return self.base_model(best_model)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        component_range = range(self.min_n_components, self.max_n_components + 1)

        model_scores = [None] * len(component_range)

        for number_of_components in component_range:
            try:
                model = self.base_model(number_of_components)
                model_score = model.score(self.X, self.lengths)
            except (ValueError, AttributeError) as e:
                continue

            other_words = set(self.words) - set(self.this_word)
            other_word_scores = [None] * len(other_words)
            for other_word in other_words:
                try:
                    other_word_scores.append(model.score(other_word, model))
                except TypeError:
                    continue
            model_penalty = sum(filter(None, other_word_scores)) / len([word for word in other_words if word])
            model_scores[number_of_components - self.min_n_components] = model_score - model_penalty

        try:
            best_model = model_scores.index(max(filter(None, model_scores))) + self.min_n_components
        except ValueError:
            return self.base_model(self.min_n_components)

        return self.base_model(best_model)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        component_range = range(self.min_n_components, self.max_n_components + 1)
        
        model_scores = [None] * len(component_range)
        
        for number_of_components in component_range:
            component_scores = []
            if len(self.sequences) > 2:
                for cv_train_idx, cv_test_idx in KFold().split(self.sequences):
                    self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    try:
                        trained_model_score = self.base_model(number_of_components).score(test_X, test_lengths)
                    except (ValueError, AttributeError) as e:
                        continue
                    component_scores.append(trained_model_score)
                if component_scores:
                    component_scores_avg = np.mean(component_scores)
                    model_scores[number_of_components - self.min_n_components] = component_scores_avg
            else:
                return self.base_model(self.min_n_components)

        if all(x is None for x in model_scores): 
            return self.base_model(self.min_n_components)

        best_model = model_scores.index(max(filter(None, model_scores))) + self.min_n_components

        return self.base_model(best_model)
