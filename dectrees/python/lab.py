import monkdata as m
import dtree


def ASSIGNMENT1():
	e1 = dtree.entropy(m.monk1)
	e2 = dtree.entropy(m.monk2)
	e3 = dtree.entropy(m.monk3)

	print("Entropy of MONK-1 Training Set:", e1)
	print("Entropy of MONK-2 Training Set:", e2)
	print("Entropy of MONK-3 Training Set:", e3)

#calcEntropy()

def ASSIGNMENT2(dataset):
	for idx, attribute in enumerate(m.attributes):
		ag = dtree.averageGain(dataset, attribute)
		print("Average gain of a{:d}: {:f}".format(idx+1, ag))

ASSIGNMENT2(m.monk1)
ASSIGNMENT2(m.monk2)
ASSIGNMENT2(m.monk3)