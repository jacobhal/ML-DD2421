import monkdata as m
import dtree
import drawtree_qt5 as draw
import random


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

def ASSIGNMENT7(iterations):
	for i in range(iterations):
		monk1train, monk1val = partition(m.monk1, 0.6)
		tree = dtree.buildTree(monk1train, m.attributes)
		dtree.allPruned(tree)
"""
	print("MONK-1")
	print('\n'.join(str(monk.identity) for monk in m.monk1))
	print("MONK-1 TRAINING SET")
	print('\n'.join(str(monk.identity) for monk in monk1train))
	print("MONK-1 VALIDATION SET")
	print('\n'.join(str(monk.identity) for monk in monk1val))
"""
