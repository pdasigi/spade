import sys
import time
import theano, numpy
from theano import tensor as T
from event_ae import EventAE
from process_data import DataProcessor 

num_args = 2
num_slots = num_args + 1
hyp_hidden_size = 50
learning_rate = 0.01
wc_hidden_sizes = [50] * num_slots
cc_hidden_sizes = [50] * num_args

dp = DataProcessor()
x_data, y_s_data, w_ind, c_ind, w_h_map = dp.make_data(sys.argv[1])
vocab_size = len(w_ind)
ont_size = len(c_ind)
event_ae = EventAE(num_args, vocab_size, ont_size, hyp_hidden_size, wc_hidden_sizes, cc_hidden_sizes)
train_func = event_ae.get_train_func(learning_rate)
for i, (x_datum, y_s_datum) in enumerate(zip(x_data, y_s_data)):
  starttime = time.time()
  train_func(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_s_datum, dtype='int32'))
  endtime = time.time()
  print >>sys.stderr, "Processed datum %d in %d seconds"%(i, endtime - starttime)

#x = T.ivector('x') # Vector of word indexes in vocabulary
#y = T.ivector('y') # Vector of concept indexes in ontology
#y_s = T.imatrix('y_s') # Matrix with all possible concept combinations
#enc_energy = event_ae.get_sym_encoder_energy(x, y)
#enc_partition = event_ae.get_sym_encoder_partition(x, y_s)
#comp_ex = event_ae.get_sym_complete_expectation(x, y_s)
#post_num = event_ae.get_sym_posterior_num(x, y)
#post_part = event_ae.get_sym_posterior_partition(x, y_s)
#rec_prob = event_ae.get_sym_rec_prob(x, y)
#f = theano.function([x, y], enc_energy)
#f1 = theano.function([x, y_s], enc_partition)
#f2 = theano.function([x, y], post_num)
#f4 = theano.function([x, y], rec_prob)
#f3 = theano.function([x, y_s], comp_ex)
#f5 = theano.function([x, y_s], post_part)
#x_g1 = numpy.asarray([0, 2, 2], dtype='int32')
#x_g2 = numpy.asarray([0, 2, 3], dtype='int32')
#x_g3 = numpy.asarray([0, 3, 4], dtype='int32')
#x_g4 = numpy.asarray([0, 4, 5], dtype='int32')
#y_g = numpy.asarray([1, 0, 0], dtype='int32')
#y_s_g = numpy.asarray([[1, 0, 0], [0, 0, 0], [0, 1, 1]], dtype='int32')
#print f(x_g1, y_g)
#print f1(x_g1, y_s_g)
#print f2(x_g1, y_g)
#print f4(x_g1, y_g)
#print f3(x_g1, y_s_g)
#print f5(x_g1, y_s_g)
#print train_func(x_g1, y_s_g)
#print train_func(x_g2, y_s_g)
#print train_func(x_g3, y_s_g)
#print train_func(x_g4, y_s_g)
#x_i, y_i = T.iscalars(2)
#f6 = theano.function([x_i, y_i], event_ae.rec_model.get_sym_rec_prob(x_i, y_i))
#for i in range(3):
#  print f6(x_g1[i], y_g[i])
