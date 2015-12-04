# A quick check to see if the representations of a random sample of synsets are closer
# to their hyponyms' representations than to a those of a randomly chosen set of non-hyponyms

import sys
import gzip
import numpy
import random
import cPickle
from nltk.corpus import wordnet as wn

max_runs = 1000
num_run = 0
n_sample_size = 10

repfile = sys.argv[1]
spade_trained = False

if len(sys.argv) > 2:
    spade_trained = True
    vocabfile = open(sys.argv[2])
    word_index = { line.split()[0] : int(line.strip().split()[1]) for line in vocabfile}    
    ontfile = open(sys.argv[3])
    ont_index = { line.split()[0] : int(line.strip().split()[1]) for line in ontfile}    

if spade_trained:
    word_rep, ont_rep = cPickle.load(open(repfile))
else:
    pt_word_rep = {(l.split()[0]).decode('utf-8'): numpy.asarray([float(f) for f in l.strip().split()[1:]]) for l in gzip.open(repfile)}


def get_prob(mean, wordvec):
    vec_m_mean = wordvec - mean
    exp_term = -0.5 * numpy.dot(vec_m_mean, vec_m_mean) * 5.0
    sqrt_term = 2 ** (0.2 * numpy.pi * len(vec_m_mean))
    return 1./numpy.sqrt(sqrt_term) * numpy.exp(exp_term)

num_less_likely_n_words = []
while num_run != max_runs:
    base_syns = []
    while base_syns == []:
        if spade_trained:
            base_word = random.sample(word_index.keys(), 1)[0]
            base_word_rep = word_rep[word_index[base_word]]
            base_syn_cands = wn.synsets(base_word)
            base_syns = []
            for bsc in base_syn_cands:
                if bsc.name() in ont_index:
                    base_syns.append(bsc)
        else:
            base_word = random.sample(pt_word_rep.keys(), 1)[0]
            base_word_rep = pt_word_rep[base_word]
            base_syns = wn.synsets(base_word)

    hyper_syns = base_syns[0].hypernyms()
    if len(hyper_syns) == 0:
        continue
    hyper_syn = hyper_syns[0]
    if spade_trained:
        if hyper_syn.name() not in ont_index:
            continue
        hyper_syn_rep = ont_rep[ont_index[hyper_syn.name()]]
    else:
        hyper_syn_hypos = hyper_syn.hyponyms()
        hyper_syn_hypo_reps = []
        hyper_syn_hypo_words = []
        for syn in hyper_syn_hypos:
            if syn == base_syns[0]:
                continue
            syn_word = syn.name().split('.')[0]
            if syn_word in pt_word_rep:
                hyper_syn_hypo_reps.append(pt_word_rep[syn_word])
                hyper_syn_hypo_words.append(syn_word)
        if len(hyper_syn_hypo_reps) < 2:
            continue
        hyper_syn_rep = numpy.mean(hyper_syn_hypo_reps, axis=0)
    n_words = []
    n_word_reps = []
    while len(n_words) < n_sample_size:
        if spade_trained:
            n_word = random.sample(word_index.keys(), 1)[0]
        else:
            n_word = random.sample(pt_word_rep.keys(), 1)[0]
        n_syns = wn.synsets(n_word)
        if len(n_syns) == 0:
            continue
        n_syn_hypernyms = [s.hypernym_paths() for s in n_syns]
        n_is_close = False
        for ll_syns in n_syn_hypernyms:
            for l_syns in ll_syns:
                for syn in l_syns:
                    if syn == base_syns[0]:
                        n_is_close = True
                        break
        if not n_is_close:
            n_words.append(n_word)
            if spade_trained:
                n_word_reps.append(word_rep[word_index[n_word]])
            else:
                n_word_reps.append(pt_word_rep[n_word])
    num_run += 1
    base_word_prob = get_prob(hyper_syn_rep, base_word_rep)
    other_word_probs = [get_prob(hyper_syn_rep, n_word_rep) for n_word_rep in n_word_reps]
    #print "Base word:", base_word
    #print "Hyper syn:", hyper_syn.name()
    #print "Hyper syn hypos:", hyper_syn_hypo_words
    #print "Other words:", n_words
    #print "Base word prob:", base_word_prob 
    #print "Other word probs:", other_word_probs
    num_lln = sum([x < base_word_prob for x in other_word_probs])
    num_less_likely_n_words.append(num_lln)

print "Probability of lower likelihood of negative samples:", float(sum(num_less_likely_n_words))/(max_runs * n_sample_size)
