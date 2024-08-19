

from random import randint
from random import random
from random import sample

sample_items = 'params3.txt'
with open(sample_items, 'rU') as kfile:
    lines = kfile.readlines()
num = int(lines[0])
cap = int(lines[num+1])
items = {int(line.split()[0]) : tuple(map(int, line.split()))[1:] for line in lines[1:num+1]} # List of possible items



max_generations = 600
K = 3
pX = 2
pU = 0.5
pM = 1.5/float(num)
P = 600
E = 5

def initialize_population():

    popS = set()
    while len(popS) < P:
        b0weight = cap+1
        while b0weight > cap:       # includes valid solutions
            b0 = tuple(randint(0,1) for _ in range(num))
            b0weight = wFitness(b0)
        popS.add(b0)
    return [list(elem) for elem in list(popS)] # set of tuples list


def mutate(chrom):  #With probability of pM, Mutation

    xman = chrom[:]
    for i in range(num):
        if pM > random():
            if chrom[i] == 0:
                xman[i] = 1
            else:
                xman[i] = 0
    return xman


def unifXover(prt1, prt2): # with probablity pX at probability pU uniform crossover for new binary lists.

    chd1 = prt1[:]
    chd2 = prt2[:]
    if pX > random():
        for i in range(num):
            #if (1/float(n)) > random():
            if pU > random():
                chd1[i] = prt2[i]
                chd2[i] = prt1[i]
    return [chd1, chd2]


def wFitness(b): # weight Fitness.

    total_weight = 0
    for idx, val in enumerate(b):
        if val == 1:
            total_weight += items[idx+1][1]
    return total_weight

def vFitness(b):  # value Fitness

    total_value = 0
    for idx, val in enumerate(b):
        if val == 1:
            total_value += items[idx+1][0]
    return total_value


def packing_info(b):  #Accepts a binary list and packed items

    indexes = []
    t_value = 0
    t_weight = 0
    for idx, val in enumerate(b):
        if val == 1:
            indexes.append(idx+1)
            t_value += items[idx+1][0]
            t_weight += items[idx+1][1]
    return [indexes, t_value, t_weight]

def tournament_selection(pop, K): # tournament  and returns a winning binary list

    tBest = 'None'
    for i in range(K):
        contestant = pop[randint(0, P-1)]
        if (tBest == 'None') or vFitness(contestant) > vFitness(tBest):
            tBest = contestant[:]
    return tBest

def select_elites(): # Selects the E best solutions a list of binary

    elites = []
    while len(elites) < E:           # choosing elites
        new_elites = populationD[max(populationD)] # binary lists with best fitness
        # If adding all elites  would be many, then discard random
        while len(new_elites) > (E - len(elites)):
            new_elites.remove(sample(new_elites, 1)[0])
        elites.extend(new_elites)
        populationD.pop(max(populationD), None) # Remove key with  value added from popD{}
    return elites

def popMean(): # Calculate the Middle fitness of current generation

    t = 0
    for i in populationR:
        t += i[0]
    return t/P

def report():
    return "max: " + str(max(populationD)) + ", middle: " + str(popMean()) + ", min: " + str(min(populationD))

def updateBest():
    return [max(populationD), populationD[max(populationD)][0], s]

def rankedList():# Make a binary list and fitness

    return [(vFitness(i), i) for i in populationL]

def rankedDict(): # Make a dictionary , keys ,fitness ,values of binary lists with that fitness

    popD = {}
    for item in populationR:
        key = item[0]
        popD.setdefault(key, []).append(item[-1])
    return popD

# initial population
populationL = initialize_population()
populationR = rankedList()
populationD = rankedDict()
s = 0 # generation counter
bestResults = updateBest()
print ("First Population  : "+str(report()))

while True:
    s += 1
    populationR = rankedList()
    populationD = rankedDict()

    # Update best
    if max(populationD) > bestResults[0]:
        bestResults = updateBest()

    # Stop  when optimum Ok
    if 'global_optimum' in globals():
        if bestResults[0] == 'global_optimum':
            print ("Global Optimum reached after "+str(s)+" generations.")
            break

    # Stop When time is max
    if s == max_generations:
        print ("GA Stopped after "+str(s)+" generations.")
        break

    # Print 10% of total progress
    if s % (max_generations / 10) == 0:
        print ("-->Best: "+str(bestResults[0])+" //// remained generations : "+str(max_generations - s)+" ////  Status: "+str(report()))

    # Start the child generation with E elites
    nextGen = select_elites()

    # Fill next generation to size P
    while len(nextGen) < P:
        parentA = tournament_selection(populationL, K)
        parentB = tournament_selection(populationL, K)
        childrenAB = unifXover(parentA, parentB)
        mutatedChildA = mutate(childrenAB[0])
        mutatedChildB = mutate(childrenAB[1])
        if (wFitness(mutatedChildA) <= cap) and (wFitness(mutatedChildB) <= cap):
            nextGen.extend([mutatedChildA, mutatedChildB])

    populationL = nextGen[:]

packing_info = packing_info(bestResults[1])
print ("\n the Best solution ("+str(bestResults[2])+"/"+str(max_generations)+"): value="+str(packing_info[1])+", weight="+str(packing_info[2])+"/"+str(cap)+"\n items:("+str(len(packing_info[0]))+"/"+str(num)+"):"+str(packing_info[0]))
