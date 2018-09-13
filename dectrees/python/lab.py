import monkdata as m
import dtree
import drawtree_qt4 as draw
import random
import numpy as np
import matplotlib.pyplot as plt


def ASSIGNMENT1():
	e1 = dtree.entropy(m.monk1)
	e2 = dtree.entropy(m.monk2)
	e3 = dtree.entropy(m.monk3)

	print("Entropy of MONK-1 Training Set:", e1)
	print("Entropy of MONK-2 Training Set:", e2)
	print("Entropy of MONK-3 Training Set:", e3)

#ASSIGNMENT1()

def ASSIGNMENT3(dataset):
	for idx, attribute in enumerate(m.attributes):
		ag = dtree.averageGain(dataset, attribute)
		print("Average gain of a{:d}: {:f}".format(idx+1, ag))

#ASSIGNMENT3(m.monk1)
#ASSIGNMENT3(m.monk2)
#ASSIGNMENT3(m.monk3)

def calcNextTreeLevel():
	selectedAttribute = m.attributes[4]
	s1 = dtree.select(m.monk1, selectedAttribute, 1)
	s2 = dtree.select(m.monk1, selectedAttribute, 2)
	s3 = dtree.select(m.monk1, selectedAttribute, 3)
	s4 = dtree.select(m.monk1, selectedAttribute, 4)

	# Calculate information gain of subsets
	#ASSIGNMENT3(s1)
	#ASSIGNMENT3(s2)
	#ASSIGNMENT3(s3)
	#ASSIGNMENT3(s4)

	mc1 = dtree.mostCommon(s1)
	mc2 = dtree.mostCommon(s2)
	mc3 = dtree.mostCommon(s3)
	mc4 = dtree.mostCommon(s4)
	#print(mc1)
	#print(mc2)
	#print(mc3)
	#print(mc4)

	tree = dtree.buildTree(m.monk2test, m.attributes)
	print(tree)
	draw.drawTree(tree)


#calcNextTreeLevel()

def ASSIGNMENT5():
	t1 = dtree.buildTree(m.monk1, m.attributes)
	print(dtree.check(t1, m.monk1test))
	print(dtree.check(t1, m.monk1))

	t2 = dtree.buildTree(m.monk2, m.attributes)
	print(dtree.check(t2, m.monk2test))
	print(dtree.check(t2, m.monk2))

	t3 = dtree.buildTree(m.monk3, m.attributes)
	print(dtree.check(t3, m.monk3test))
	print(dtree.check(t3, m.monk3))

#ASSIGNMENT5()

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def prune(currentRatio, tree, validationSet):
	pruningCandidates = dtree.allPruned(tree)
	ratios = list(map(lambda lst: dtree.check(lst, validationSet), pruningCandidates))
	#print(ratios)
	maxR = max(ratios)
	maxI = ratios.index(max(ratios))
	#print("Current is: {:f}".format(currentRatio))
	if currentRatio < maxR:
		#print("Found new max: {:f}".format(maxR))
		return prune(maxR, pruningCandidates[maxI], validationSet)
	else:
		return float(currentRatio)

def pruningTest(dataset, fraction): # returns the error classification ratio 
		monktrain, monkval = partition(dataset, fraction)
		tree = dtree.buildTree(monktrain, m.attributes)
		curRatio = dtree.check(tree, monkval)			
		maxR = prune(curRatio, tree, monkval)
		#print("Max is: {:f}".format(maxR))
		return 1-maxR

#pruningTest(m.monk1, 0.6)

def ASSIGNMENT7(dataset, iterations):
	fractions = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
	sumAverage = []
	for fraction in fractions:
		for itr in range(iterations):
			sumAverage.append(pruningTest(dataset, fraction))
		n, bins, patches = plt.hist(sumAverage, 50, normed=1, facecolor='green', alpha=0.75)
		plt.xlabel("Error ratio")
		plt.ylabel("Density")
		plt.title("Measure of spread for MONK-1 dataset according to error ratios over " + iterations + " iteratons")
		plt.axis([0,1, 0, iterations])
		plt.grid(True)

		plt.show()
		#print("Mean error of dataset using fraction = {:f} and {:d} iterations: {:f}".format(fraction, iterations, sumAverage))

ASSIGNMENT7(m.monk1, 100)

"""
	print("MONK-1")
	print('\n'.join(str(monk.identity) for monk in m.monk1))
	print("MONK-1 TRAINING SET")
	print('\n'.join(str(monk.identity) for monk in monk1train))
	print("MONK-1 VALIDATION SET")
	print('\n'.join(str(monk.identity) for monk in monk1val))
"""
