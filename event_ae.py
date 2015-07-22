import theano, numpy
from theano import tensor as T

from hypernymy import HypernymModel
from preferences import PreferenceModel
from reconstruction import ReconstructionModel 

class EventAE(object):
	def __init__(self, num_args, vocab_size, ont_size, hyp_hidden_size, wc_hidden_sizes, cc_hidden_sizes, word_dim=50, concept_dim=50):
		numpy_rng = numpy.random.RandomState(12345)
		vocab_rep_range = 4 * numpy.sqrt(6. / (vocab_size + word_dim))
		init_vocab_rep = numpy.asarray(numpy_rng.uniform(low = -vocab_rep_range, high = vocab_rep_range, size=(vocab_size, word_dim)) )
		ont_rep_range = 4 * numpy.sqrt(6. / (ont_size + concept_dim))
		init_ont_rep = numpy.asarray(numpy_rng.uniform(low = -ont_rep_range, high = ont_rep_range, size=(ont_size, concept_dim)) )
		vocab_rep = theano.shared(value=init_vocab_rep, name='vocab_rep')
		ont_rep = theano.shared(value=init_ont_rep, name='ont_rep')
		self.hyp_model = HypernymModel('linlayer', hyp_hidden_size, vocab_rep, ont_rep)
		self.wc_pref_models = []
		self.cc_pref_models = []
		for i in range(num_slots):
			self.wc_pref_models.append(PreferenceModel('word_concept', 'linlayer', wc_hidden_sizes[i], ont_rep, vocab_rep))
	
		for i in range(num_args):
			self.cc_pref_models.append(PreferenceModel('concept_concept', 'linlayer', cc_hidden_sizes[i], ont_rep))
		self.rec_model = ReconstructionModel(ont_size, vocab_rep)
		self.num_args = num_args
		self.num_slots = self.num_args + 1 # +1 for the predicate

	def get_sym_encoder_energy(self, x, y):
		hsum = T.constant(0)
		for i in range(num_slots):
			hsum += self.hyp_model.get_symb_score(x[i], y[i])
		p_w_c_sum = T.constant(0)
		for i in range(num_slots):
			for j in range(num_slots):
				if i == j:
					continue
				p_w_c_sum += self.wc_pref_models[i].get_symb_score(x[i], y[j])
		p_c_c_sum = T.constant(0)
		for i in range(num_args):
			p_c_c_sum += self.cc_pref_models[i].get_symb_score(y[0], y[i])
		return hsum + p_w_c_sum + p_c_c_sum

	def get_sym_encoder_partition(self, x, y_s):
		partial_sums, _ = theano.scan(fn=lambda y, interm_sum, x_0: interm_sum + T.exp(self.get_sym_encoder_energy(x_0, y)), outputs_info=numpy.asarray(0.0, dtype='float64'), sequences=[y_s], non_sequences=x)
		encoder_partition = partial_sums[-1]
		return encoder_partition

	def get_sym_rec_prob(self, x, y):
		#rec_prob = T.constant(1.0)
		#for i in range(num_slots):
		#	_, p_r = self.rec_model.get_sym_rec_prob(x[i], y[i])
		#	rec_prob *= p_r
		init_prob = T.constant(1.0, dtype='float64')
		partial_prods, _ = theano.scan(fn = lambda x_i, y_i, interm_prod: interm_prod * self.rec_model.get_sym_rec_prob(x_i, y_i), outputs_info=init_prob, sequences=[x, y])
		rec_prob = partial_prods[-1]
		return rec_prob
		
	def get_sym_posterior_num(self, x, y):
		enc_energy = self.get_sym_encoder_energy(x, y)
		rec_prob = self.get_sym_rec_prob(x, y)
		return T.exp(enc_energy) * rec_prob
		#return T.exp(enc_energy)
			
	def get_sym_posterior_partition(self, x, y_s):
		partial_sums, _ = theano.scan(fn=lambda y, interm_sum, x_0: interm_sum + self.get_sym_posterior_num(x_0, y), outputs_info=numpy.asarray(0.0, dtype='float64'), sequences=[y_s], non_sequences=x)
		posterior_partition = partial_sums[-1]
		return posterior_partition

	def get_sym_complete_expectation(self, x, y_s):
		encoder_partition = self.get_sym_encoder_partition(x, y_s)
		posterior_partition = self.get_sym_posterior_partition(x, y_s)
		prod_fun = lambda y_0, interm_sum, x_0: interm_sum + \
				self.get_sym_posterior_num(x_0, y_0) * \
				( self.get_sym_encoder_energy(x_0, y_0) - T.log(encoder_partition) + \
					T.log(self.get_sym_rec_prob(x_0, y_0)) )
		partial_sums, _ = theano.scan(fn=prod_fun, outputs_info=numpy.asarray(0.0, dtype='float64'), sequences=[y_s], non_sequences=x)
		complete_expectation = partial_sums[-1] / posterior_partition
		return complete_expectation
		

num_args = 2
num_slots = num_args + 1
hyp_hidden_size = 2
vocab_size = 50
ont_size = 20
wc_hidden_sizes = [2] * num_slots
cc_hidden_sizes = [2] * num_args
event_ae = EventAE(num_args, vocab_size, ont_size, hyp_hidden_size, wc_hidden_sizes, cc_hidden_sizes)

x = T.ivector('x') # Vector of word indexes in vocabulary
y = T.ivector('y') # Vector of concept indexes in ontology
y_s = T.imatrix('y_s') # Matrix with all possible concept combinations
enc_energy = event_ae.get_sym_encoder_energy(x, y)
enc_partition = event_ae.get_sym_encoder_partition(x, y_s)
comp_ex = event_ae.get_sym_complete_expectation(x, y_s)
post_num = event_ae.get_sym_posterior_num(x, y)
post_part = event_ae.get_sym_posterior_partition(x, y_s)
rec_prob = event_ae.get_sym_rec_prob(x, y)
f = theano.function([x, y], enc_energy)
f1 = theano.function([x, y_s], enc_partition)
f2 = theano.function([x, y], post_num)
f4 = theano.function([x, y], rec_prob)
f3 = theano.function([x, y_s], comp_ex)
f5 = theano.function([x, y_s], post_part)
print f(numpy.asarray([0, 2, 2], dtype='int32'), numpy.asarray([1, 0, 0], dtype='int32'))
print f1(numpy.asarray([0, 2, 2], dtype='int32'), numpy.asarray([[1, 0, 0], [0, 0, 0], [0, 1, 1]], dtype='int32'))
print f2(numpy.asarray([0, 2, 2], dtype='int32'), numpy.asarray([1, 0, 0], dtype='int32'))
print f4(numpy.asarray([0, 2, 2], dtype='int32'), numpy.asarray([1, 0, 0], dtype='int32'))
print f3(numpy.asarray([0, 2, 2], dtype='int32'), numpy.asarray([[1, 0, 0], [0, 0, 0], [0, 1, 1]], dtype='int32'))
print f5(numpy.asarray([0, 2, 2], dtype='int32'), numpy.asarray([[1, 0, 0], [0, 0, 0], [0, 1, 1]], dtype='int32'))
