import theano, numpy
from theano import tensor as T
from event_ae import EventAE

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
