import sys
import codecs
import gzip
import argparse
import cPickle
import theano
import numpy
from theano import tensor as T

from event_ae import EventAE
from process_data import DataProcessor

argparser = argparse.ArgumentParser(description="Score using Selectional Preference AutoencoDEr")
argparser.add_argument('test_file', metavar="TEST-FILE", type=str, help="File containing n-tuples to score")
argparser.add_argument('word_types', metavar="WORD-TYPES", type=str, help="String showing the WordNet POS types of the words in the train file, separated by _. Eg. a_n for adj-noun, v_n_n for verb-noun-noun etc.")
argparser.add_argument('--vocab_file', type=str, help="Word vocabulary file", default="vocab.txt")
argparser.add_argument('--ont_file', type=str, help="Concept vocabulary file", default="ont.txt")
argparser.add_argument('--use_relaxation', help="Ignore inter-concept preferences and optimize", action='store_true')
argparser.set_defaults(use_relaxation=False)
#argparser.add_argument('--pt_rep', type=str, help="File containing pretrained embeddings")
argparser.add_argument('--use_em', help="Use EM (Default is False)", action='store_true')
argparser.set_defaults(use_em=False)
argparser.add_argument('--use_nce', help="Use NCE for estimating encoding probability. (Default is False)", action='store_true')
argparser.set_defaults(use_nce=False)
argparser.add_argument('--hyp_model_type', type=str, help="Hypernymy model (weighted_prod, linlayer, tanhlayer)", default="weighted_prod")
argparser.add_argument('--wc_pref_model_type', type=str, help="Word-concept preference model (weighted_prod, linlayer, tanhlayer)", default="tanhlayer")
argparser.add_argument('--cc_pref_model_type', type=str, help="Concept-concept preference model (weighted_prod, linlayer, tanhlayer)", default="tanhlayer")
argparser.add_argument('--param_iter', type=int, help="Iteration of learned param to use (default 1)", default=1)
args = argparser.parse_args()

use_relaxation = args.use_relaxation
pred_arg_pos = args.word_types.split("_")
dp = DataProcessor(pred_arg_pos)
x_data, y_s_data, w_ind, c_ind, _, _, _ = dp.make_data(args.test_file, relaxed=args.use_relaxation, handle_oov=False)

num_slots = len(pred_arg_pos)
num_args = num_slots - 1
hyp_hidden_size = 20
wc_hidden_sizes = [20] * num_slots
cc_hidden_sizes = [20] * num_args

#use_pretrained_wordrep = False
#if args.pt_rep:
#  print >>sys.stderr, "Using pretrained word representations from %s"%(args.pt_rep)
#  use_pretrained_wordrep = True
#  pt_word_rep = {l.split()[0]: numpy.asarray([float(f) for f in l.strip().split()[1:]]) for l in gzip.open(args.pt_rep)}

train_vocab_file = codecs.open(args.vocab_file, "r", "utf-8")
train_ont_file = codecs.open(args.ont_file, "r", "utf-8")
vocab_rep, ont_rep = cPickle.load(open("repr_params_%d.pkl"%args.param_iter, "rb"))
hyp_params = cPickle.load(open("hyp_params_%d.pkl"%args.param_iter, "rb"))
wcp_params = cPickle.load(open("wcp_params_%d.pkl"%args.param_iter, "rb"))
if not use_relaxation:
  ccp_params = cPickle.load(open("ccp_params_%d.pkl"%args.param_iter, "rb"))
rec_params = cPickle.load(open("rec_params_%d.pkl"%args.param_iter, "rb"))

_, word_dim = vocab_rep.shape
word_rep_min, word_rep_max = vocab_rep.min(), vocab_rep.max()
_, conc_dim = ont_rep.shape
conc_rep_min, conc_rep_max = ont_rep.min(), ont_rep.max()
numpy_rng = numpy.random.RandomState(12345)

train_vocab_map = {}
rev_train_vocab_map = {}
train_ont_map = {}
rev_train_ont_map = {}

for line in train_vocab_file:
  w, ind = line.strip().split()
  train_vocab_map[w] = int(ind)
  rev_train_vocab_map[int(ind)] = w
for line in train_ont_file:
  c, ind = line.strip().split()
  train_ont_map[c] = int(ind)
  rev_train_ont_map[int(ind)] = c

vocab_size = len(train_vocab_map)
ont_size = len(train_ont_map)

test_train_vocab_map = {}
test_train_ont_map = {}
ignored_c_inds = set([])

new_words = []
#if use_pretrained_wordrep:
#  num_new_words_found = 0

for w in w_ind:
  if w in train_vocab_map:
    train_ind = train_vocab_map[w]
  else:
    new_words.append(w)
    #if use_pretrained_wordrep:
    #  if w in pt_word_rep:
    #    num_new_words_found += 1
    #    w_rep = numpy.asarray([pt_word_rep[w]])
    #  else:
    #    w_rep = numpy_rng.uniform(low = word_rep_min, high = word_rep_max, size=(1,word_dim))
    #else:
    #w_rep = numpy_rng.uniform(low = word_rep_min, high = word_rep_max, size=(1,word_dim))
    #vocab_rep = numpy.concatenate([vocab_rep, w_rep])
    #vocab_size += 1
    #train_ind = vocab_size - 1
    train_ind = 0 # for UNK
  #train_ind = train_vocab_map[w] if w in train_vocab_map else train_vocab_map["pele"]
  test_train_vocab_map[w_ind[w]] = train_ind

#num_new_words = len(new_words)
print >>sys.stderr, "New words in test set: %s"%new_words
#if num_new_words > 0 and use_pretrained_wordrep:
#  print >>sys.stderr, "%.2f found in pretrained rep"%(float(num_new_words_found)/num_new_words)

new_concepts = []
for c in c_ind:
  if c in train_ont_map:
    train_ind = train_ont_map[c]
  else:
    new_concepts.append(c)
    #c_rep = numpy_rng.uniform(low = conc_rep_min, high = conc_rep_max, size=(1,conc_dim))
    #ont_rep = numpy.concatenate([ont_rep, c_rep])
    #ont_size += 1
    #train_ind = ont_size - 1
    train_ind = 0 # for UNK
  test_train_ont_map[c_ind[c]] = train_ind
num_new_concepts = len(new_concepts)
print >>sys.stderr, "New concepts in test set: %s"%new_concepts

fixed_data = []

ignored_y_cands = 0

for x_datum, y_s_datum in zip(x_data, y_s_data):
  fixed_x = [test_train_vocab_map[ind] for ind in x_datum[:-1]] + [x_datum[-1]] if args.use_relaxation else [test_train_vocab_map[ind] for ind in x_datum]
  fixed_y_s = []
  if args.use_relaxation:
    for y_datum in y_s_datum:
      if y_datum not in test_train_ont_map:
        ignored_y_cands += 1
        continue
      fixed_y_s.append(test_train_ont_map[y_datum])
  else:
    for y_datum in y_s_datum:
      fixed_y = []
      all_present = True
      for y_ind in y_datum:
        if y_ind not in test_train_ont_map:
          all_present = False
          ignored_y_cands += 1
          break
        fixed_y.append(test_train_ont_map[y_ind])
      if all_present:
        fixed_y_s.append(fixed_y)
  #fixed_y_s = [[test_train_ont_map[ind] for ind in y_datum] for y_datum in y_s_datum]
  fixed_data.append((fixed_x, fixed_y_s))

print >>sys.stderr, "Ignored an average of %f y cands per data point"%(float(ignored_y_cands)/len(fixed_data))
print >>sys.stderr, len(x_data), len(y_s_data), len(fixed_data)


event_ae = EventAE(num_args, vocab_size, ont_size, hyp_hidden_size, wc_hidden_sizes, cc_hidden_sizes, relaxed=use_relaxation, hyp_model_type=args.hyp_model_type, wc_pref_model_type=args.wc_pref_model_type, cc_pref_model_type=args.cc_pref_model_type)
#for i, param in enumerate(repr_params):
#  event_ae.repr_params[i].set_value(param)
event_ae.vocab_rep.set_value(vocab_rep)
event_ae.ont_rep.set_value(ont_rep)

event_ae.hyp_model.set_params(hyp_params)
for i, param_dict in enumerate(wcp_params):
  for j in param_dict:
    event_ae.wc_pref_models[i][j].set_params(param_dict[j])
if not use_relaxation:
  for i, params in enumerate(ccp_params):
    event_ae.cc_pref_models[i].set_params(params)
event_ae.rec_model.set_params(rec_params)

x = T.ivector('x')
if args.use_relaxation:
  y_s = T.ivector('y_s')
  comp_prob_funcs = [theano.function([x, y_s], T.log(event_ae.get_sym_relaxed_direct_prob(x, y_s, s))) for s in range(num_slots)]
else:
  y_s = T.imatrix('y_s')
  sym_comp_prob = T.log(event_ae.get_sym_nc_direct_prob(x, y_s)) if args.use_nce else T.log(event_ae.get_sym_direct_prob(x, y_s))
  comp_prob_func = theano.function([x, y_s], sym_comp_prob)

def get_comp_prob(x_datum, y_s_datum):
  return comp_prob_func(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_s_datum, dtype='int32'))

def get_relaxed_comp_prob(x_data, y_s_data):
  # This function expects num_slots * (num_slots - 1) datapoints to calculate the comp_prob for the entire predarg structure
  logprobs = []
  for i in range(num_slots):
    x_datum = x_data[i]
    cprob_func = comp_prob_funcs[i]
    y_s_datum = y_s_data[i]
    if len(y_s_datum) == 0:
      logprobs.append(-float("inf"))
      break
    else:
      logprobs.append(cprob_func(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_s_datum, dtype='int32')))
  return logprobs

if args.use_relaxation:
  post_score_funcs = [event_ae.get_relaxed_posterior_func(s) for s in range(num_slots)]
else:
  post_score_func = event_ae.get_posterior_func()

def get_mle_y(x_datum, y_s_datum):
  max_score = -float("inf")
  best_y = []
  for y_datum in y_s_datum:
    if args.use_relaxation:
      s = x_datum[-1]
      pscore_func = post_score_funcs[s]
      score = numpy.log(pscore_func(numpy.asarray(x_datum[:-1], dtype='int32'), numpy.asarray(y_datum, dtype='int32')))
    else:
      global post_score_func
      score = post_score_func(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_datum, dtype='int32'))
    if score > max_score:
      max_score = score
      best_y = y_datum
  return best_y, max_score


if args.use_relaxation:
  for i in range(0, len(fixed_data), num_slots):
    x_data = [x_datum for x_datum, _ in fixed_data[i:i+num_slots]]
    y_s_data = [y_s_datum for _, y_s_datum in fixed_data[i:i+num_slots]]
    mle_y_out = [get_mle_y(x_d, y_s_d) for x_d, y_s_d in zip(x_data, y_s_data)]
    mle_ys, mle_y_scores = zip(*mle_y_out)
    mle_y_words = [rev_train_ont_map[ind] for ind in mle_ys]
    #print >>sys.stderr, x_data, y_s_data
    logprobs = get_relaxed_comp_prob(x_data, y_s_data)
    comp_prob = sum(logprobs)
    mle_prob = sum(mle_y_scores)
    print comp_prob, (" ".join(mle_y_words)).encode("utf-8")
    #print " ".join(str(f) for f in logprobs), (" ".join(mle_y_words)).encode("utf-8")
    #print mle_prob
else:
  for x_datum, y_s_datum in fixed_data:
    comp_prob = get_comp_prob(x_datum, y_s_datum) if len(y_s_datum) != 0 else -float("inf")
    mle_y, y_score = get_mle_y(x_datum, y_s_datum)
    mle_y_words = [rev_train_ont_map[ind] for ind in mle_y]
    pred_arg = [rev_train_vocab_map[ind] for ind in x_datum]
    print comp_prob, (" ".join(pred_arg)).encode('utf-8'), (" ".join(mle_y_words)).encode('utf-8')


