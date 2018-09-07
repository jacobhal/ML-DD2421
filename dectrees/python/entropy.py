import monkdata as m
import dtree


def calcEntropy():
	e1 = dtree.entropy(m.monk1)
	e2 = dtree.entropy(m.monk2)
	e3 = dtree.entropy(m.monk3)

	print("Entropy of MONK-1 Training Set:", e1)
	print("Entropy of MONK-2 Training Set:", e2)
	print("Entropy of MONK-3 Training Set:", e3)

calcEntropy()