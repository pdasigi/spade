import numpy, theano
from theano import tensor as T

# TODO: Covariance parameters may be too simple. Think of a way to make the matrix positive semi-definite and non-singular

class ReconstructionModel(object):
  def __init__(self, ont_size, vocab_rep):
    """
    ont_size (int):  Number of concepts in the ontology
    word_dim (int):  Dimensionality of word representations
    """
    numpy_rng = numpy.random.RandomState(12345)
    _, word_dim = vocab_rep.get_value().shape
    self.vocab_rep = vocab_rep
    avg_range = 4 * numpy.sqrt(6. / (ont_size + word_dim))
    init_avgs = numpy.asarray(numpy_rng.uniform(low = -avg_range, high = avg_range, size = (ont_size, word_dim)))
    self.avgs = theano.shared(value = init_avgs, name = 'avgs')
    init_cov_multiples = numpy.asarray([0.2]*ont_size)
    self.cov_multiples = theano.shared(value = init_cov_multiples, name = 'cov_mult')
    self.word_dim = word_dim

  def get_sym_rec_prob(self, word_ind, concept_ind):
    avg, cov_m = self.avgs[concept_ind], self.cov_multiples[concept_ind]
    # TODO: Temporary change to fix covariance at 1.  Change this later
    #cov_m = 0 * cov_m + 0.2
    word_rep = self.vocab_rep[word_ind]
    rep_m_avg = word_rep - avg
    exp_term = -0.5 * T.dot(rep_m_avg, rep_m_avg) * (1. / T.abs_(cov_m))
    sqrt_term = T.pow(2 * T.abs_(cov_m) * numpy.pi, self.word_dim)
    self.p_r = 1. / T.sqrt(sqrt_term) * T.exp(exp_term)
    return self.p_r

  def get_params(self):
    return [self.avgs, self.cov_multiples]

  def set_params(self, params):
    avgs, cov_multiples = params
    self.avgs.set_value(avgs)
    self.cov_multiples.set_value(cov_multiples)
