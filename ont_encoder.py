from nltk.corpus import wordnet as wn
import gzip
import numpy
import theano
from theano import tensor as T
import sys

class OntologyEncoder(object):
  def __init__(self, repfile):
    self.numpy_rng = numpy.random.RandomState(12345)
    self.pt_word_rep = {}
    self.all_rep_min = float("inf")
    self.all_rep_max = -float("inf")
    for l in gzip.open(repfile):
      l_parts = l.strip().split()
      rep = numpy.asarray([float(f) for f in l_parts[1:]])
      rep_min, rep_max = min(rep), max(rep)
      if rep_min < self.all_rep_min:
        self.all_rep_min = rep_min
      if rep_max > self.all_rep_max:
        self.all_rep_max = rep_max
      self.pt_word_rep[l_parts[0]] = rep
    self.rep_dim = len(rep)
  
  def encode_ont(self, syn_names=None):
    leaves = []
    non_leaves = []
    non_leaf_hyps = []
    if syn_names is not None:
      all_syns = []
      for syn_name in syn_names:
        if len(syn_name.split('.')) == 3:
          all_syns.append(wn.synset(syn_name))
        else:
          all_syns.append(syn_name)
    else:
      all_syns = wn.all_synsets()

    for syn in all_syns:
      if type(syn) is str or type(str) is unicode:
        leaves.append(syn)
      else:
        syn_hyponyms = syn.hyponyms()
        if len(syn_hyponyms) == 0:
          leaves.append(syn)
        else:
          non_leaves.append(syn)
          non_leaf_hyps.append((syn, syn_hyponyms))

    print >>sys.stderr, "Leaves:", len(leaves)
    print >>sys.stderr, "Non leaves:", len(non_leaves)

    known_set = set(leaves).union(set(non_leaves))
    leaf_non_leaves = []
    real_non_leaves = []
    single_hyp_nls = []
    for nl, nl_hyps in non_leaf_hyps:
      known_hyps = set(nl_hyps).intersection(known_set)
      if len(known_hyps) == 0:
        leaf_non_leaves.append(nl)
      else:
        if len(known_hyps) == 1:
          single_hyp_nls.append(nl)
        real_non_leaves.append((nl, known_hyps))

    print >>sys.stderr, "Non leaves that are practically leaves: %d"%(len(leaf_non_leaves))
    print >>sys.stderr, "Non leaves that have just one known hyponym: %d"%(len(single_hyp_nls))

    num_reps_per_syn = []
    num_zero_rep_syns = 0

    num_leaves = len(leaves) + len(leaf_non_leaves)
    init_shared_reps = self.numpy_rng.uniform(low=self.all_rep_min, high=self.all_rep_max, size=(num_leaves, self.rep_dim))
    leaf_syn_index = {}

    for syn in leaves + leaf_non_leaves:
      syn_name = syn if type(syn) is str or type(str) is unicode else syn.name()
      leaf_syn_index[syn_name] = len(leaf_syn_index)
      lemma_reps = []
      if not (type(syn) is str or type(syn) is unicode):
        lemmas = [x.lower() for x in syn.lemma_names()]
        for lemma in lemmas:
          if lemma in self.pt_word_rep:
            lemma_reps.append(self.pt_word_rep[lemma])
      num_lemma_reps = len(lemma_reps)
      num_reps_per_syn.append(num_lemma_reps)
      if num_lemma_reps != 0:
        init_shared_reps[leaf_syn_index[syn.name()]] = numpy.mean(lemma_reps, axis=0)
      else:
        num_zero_rep_syns += 1

    leaf_reps = theano.shared(init_shared_reps, name='leaf_reps')
    non_leaf_reps = []

    non_leaf_syn_index = {}

    # Even though a node is a nonleaf, it can still have leaf hyponyms.

    while len(real_non_leaves) != len(non_leaf_reps):
      num_unrepresented_nls = 0
      for nl, known_nl_hyps in real_non_leaves:
        nl_name = nl if type(nl) is str or type(nl) is unicode else nl.name()
        if nl_name in non_leaf_syn_index:
          continue
        nl_hyp_reps = []
        unrepresented = False
        for hyp in known_nl_hyps:
          hyp_name = hyp if type(hyp) is str or type(hyp) is unicode else hyp.name()
          if hyp_name in leaf_syn_index:
            nl_hyp_reps.append(leaf_reps[leaf_syn_index[hyp_name]])
          elif hyp.name() in non_leaf_syn_index:
            nl_hyp_reps.append(non_leaf_reps[non_leaf_syn_index[hyp_name]])
          else:
            num_unrepresented_nls += 1
            unrepresented = True
            break
        if unrepresented:
          continue
        non_leaf_syn_index[nl_name] = len(non_leaf_syn_index)
        nl_rep = T.mean(nl_hyp_reps, axis=0)
        non_leaf_reps.append(nl_rep)
      print "Unrepresented: %d"%(num_unrepresented_nls)

    ont_index = {}
    ont_rep = []

    for syn_name in syn_names:
      ont_index[syn_name] = len(ont_index)
      if syn_name in leaf_syn_index:
        ont_rep.append(leaf_reps[leaf_syn_index[syn_name]])
      else:
        ont_rep.append(non_leaf_reps[non_leaf_syn_index[syn_name]])

    ont_rep = theano.shared(ont_rep)

    print >>sys.stderr, "No reps for %d synsets"%(num_zero_rep_syns)
    print >>sys.stderr, "Average lemmas with reps per syn: %f"%(float(sum(num_reps_per_syn))/len(num_reps_per_syn))
    return leaf_reps, ont_rep, ont_index

if __name__ == '__main__':
  syn_names = [x.split()[0] for x in open(sys.argv[2]).readlines()]
  oe = OntologyEncoder(sys.argv[1])
  leafreps, ontreps, ont_index = oe.encode_ont(syn_names)
  for i, syn_name in enumerate(syn_names):
    assert ont_index[syn_name] == i
  print len(ont_index)
