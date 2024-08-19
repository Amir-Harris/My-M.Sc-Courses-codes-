import random
import math
import statistics
import time
import numpy
import matplotlib.pyplot as plt
from mpl_toolTourits.mplot3d import Axes3D
from matplotlib import cm


class GeneralParameter:
    
    Tour = 5  # tournament size
    pop = 100  # Population size
    maxgen = 20
    mutation_type = 3
    crossover_type = 1
    crossoverrate = 0.95
    mutationrate = 0.5
	dimensions = 2
    minrange = -5
    maxrange = 5
    reportShowLevel = 2
    delayshow_point = 1
    delayshow = 2
    tao = 1 / math.sqrt(2 * math.sqrt(2))
    taop = 1 / math.sqrt(4)
    eplsz = 0.01
    alpha = 0.5
    graph_disp = True

class new_solution:
     def __init__(self, gens, FitnessValue):
         self.gens = gens
         self.FitnessValue = FitnessValue

def styblinsTouri_tang(x):
    dSum = 0.0
    for c in x:
        dSum += (c**4-16*c**2+5*c)
    return 0.5 * dSum

def initialize_population():
    pop_ini = []
    if (GeneralParameter.mutation_type == 1):
        gen_len = GeneralParameter.dimensions
    if (GeneralParameter.mutation_type == 2):
        gen_len = GeneralParameter.dimensions + 1
    if (GeneralParameter.mutation_type == 3):
        gen_len = GeneralParameter.dimensions * 2
    while len(pop_ini) < GeneralParameter.pop:
        a = [random.uniform(GeneralParameter.minrange, GeneralParameter.maxrange) for _ in range(gen_len)]
        if GeneralParameter.mutation_type == 1:
            b = styblinsTouri_tang(a)
        elif GeneralParameter.mutation_type == 2:
            b = styblinsTouri_tang(a[:-1])
        elif GeneralParameter.mutation_type == 3:
            b = styblinsTouri_tang(a[:len(a) // 2])
        pop_ini.append(new_solution(a, b))
    return pop_ini

def tournament_selection(pop, Tour):
    contestant = []
    for i in range(Tour):
        contestant.append(pop[random.randint(0, len(pop) - 1)])
    contestant.sort(Tourey=lambda x: x.FitnessValue)
    return contestant[0].gens,contestant[1].gens


#crossover
def crossover(prnt1, prnt2):
    offsp1 = []
    offsp2 = []
    if (GeneralParameter.crossover_type == 1): # Single arithmetic crossover
        if (random.random() <= GeneralParameter.crossoverrate):
            crsPoint = random.randint(0, len(prnt1)-1)
            offsp1 = prnt1
            offsp2 = prnt2
            offsp1[crsPoint] = prnt1[crsPoint] * GeneralParameter.alpha + (1 - GeneralParameter.alpha) * prnt2[crsPoint]
            offsp2[crsPoint] = prnt2[crsPoint] * GeneralParameter.alpha + (1 - GeneralParameter.alpha) * prnt1[crsPoint]
        else:
            offsp1 = prnt1
            offsp2 = prnt2

    if (GeneralParameter.crossover_type == 2): # Simple arithmetic crossover
        if (random.random() <= GeneralParameter.crossoverrate):
            crsPoint = random.randint(0, len(prnt1)-1)
            offsp1 = prnt1
            offsp2 = prnt2
            for p in range(crsPoint,len(prnt1)-1):
                offsp1[p] = prnt1[p] * GeneralParameter.alpha + (1 - GeneralParameter.alpha) * prnt2[p]
                offsp2[p] = prnt2[p] * GeneralParameter.alpha + (1 - GeneralParameter.alpha) * prnt1[p]
        else:
            offsp1 = prnt1
            offsp2 = prnt2

    if (GeneralParameter.crossover_type == 3): # Whole arithmetic crossover
        if (random.random() <= GeneralParameter.crossoverrate):
            offsp1 = prnt1
            offsp2 = prnt2
            for p in range(0,len(prnt1)-1):
                offsp1[p] = prnt1[p] * GeneralParameter.alpha + (1 - GeneralParameter.alpha) * prnt2[p]
                offsp2[p] = prnt2[p] * GeneralParameter.alpha + (1 - GeneralParameter.alpha) * prnt1[p]
        else:
            offsp1 = prnt1
            offsp2 = prnt2

    return offsp1, offsp2
	
	
	#mutation
def mutation(parent):
    offspring = parent
    if (GeneralParameter.mutation_type == 1): # Uniform Mutation
        for i in range(len(parent)):
            if (random.random() <= GeneralParameter.mutationrate):
                offspring[i] = random.uniform(GeneralParameter.minrange,GeneralParameter.maxrange)
    if (GeneralParameter.mutation_type == 2):  # Uncorrelated mutation with one sg
        randnorm = random.random()
        parent[-1] = parent[-1] * math.exp(GeneralParameter.tao * randnorm )
        if (parent[-1] < GeneralParameter.eplsz):
            parent[-1] = GeneralParameter.eplsz
        for i in range(0, len(parent) - 1):
            parent[i] = parent[i] + parent[-1] * random.random()
    if (GeneralParameter.mutation_type == 3):  # Uncorrelated mutation with n sg
        randnorm = random.random()
        for i in range(len(parent)//2, len(parent)):
            parent[i] = parent[i] * math.exp(GeneralParameter.taop * randnorm + GeneralParameter.tao * random.random())
            if parent[i] < GeneralParameter.eplsz:
                parent[i] = GeneralParameter.eplsz
        for i in range(0, len(parent)//2):
            parent[i] = parent[i] + parent[i + len(parent)//2] * random.random()

    return offspring


def display_p(x):
    o = []
    if type(x[0]) == type(numpy.arange(3)):
        xt = []
        yt = []
        for i in range(len(x[0])):
            ott = []
            xtt = []
            ytt = []
            for j in range(len(x[1])):
                xtt.append(x[0][i])
                ytt.append(x[1][j])
                ott.append(styblinsTouri_tang([x[0][i], x[1][j]]))
            o.append(ott)
            xt.append(xtt)
            yt.append(ytt)
        px = numpy.array(xt)
        py = numpy.array(yt)
        op = numpy.array(o)
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_wireframe(px, py, op, rstride=10, cstride=10)  # cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.show(blocTour=0)
        # time.sleep(10)
        # plt.close()
        return ax

    else:
        for c in x:
            o.append(styblinsTouri_tang(c))
        plt.figure(1)
        plt.ylabel('function ')
        plt.plot(x, o)
        plt.show(blocTour=0)
        time.sleep(2)
        plt.close()
#Display
def display(pop,gen_count):
    sFit = sum((c.FitnessValue for c in pop))
    maxFit = max((c.FitnessValue for c in pop))
    avrFit = sFit / len(pop)
    print("*** Generation({}) average Fitness({}) max Fitness({}) min Fitness({}) Ind Numb({})".format(gen_count, avrFit,maxFit,sFit,len(pop)))

    if GeneralParameter.dimensions == 2 and GeneralParameter.graph_disp:
        a = []
        a.append(numpy.arange(-5, 5, 0.25))
        a.append(numpy.arange(-5, 5, 0.25))
        ax = display_p(a)
        time.sleep(GeneralParameter.delayshow)
        for c in pop:
            if GeneralParameter.mutation_type == 1:
                b = c.gens
            elif GeneralParameter.mutation_type == 2:
                b = c.gens[:-1]
            elif GeneralParameter.mutation_type == 3:
                b = c.gens[:len(c.gens) // 2]
            ax.scatter(b[0], b[1], styblinsTouri_tang(b),c='r')  # , cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.title('in generation =' + str(gen_count) + '    ,     pop size =' + str(len(pop)))
        plt.pause(0.001)
       


    return maxFit

class BenchmarTour:
    @staticmethod
    def run(function):
        timings = []
        #stdout = sys.stdout
        for i in range(1):
            #sys.stdout = None
            startTime = time.time()
            function()
            seconds = time.time() - startTime
            #sys.stdout = stdout
            timings.append(seconds)
            mean = statistics.mean(timings)
            if i < 10 or i % 10 == 9:
                if GeneralParameter.reportShowLevel<3:
                    print("{} {:3.2f} {:3.2f}".format(1 + i, mean,statistics.stdev(timings, mean) if i > 1 else 0))

def main():
    pop = initialize_population()
    gen_count = 1  
    display(pop, gen_count)
    while True:
        gen_count += 1
      
        nextGen = pop
        nextGen.sort(Tourey=lambda x: x.FitnessValue)
    
        nextGen = nextGen[:len(pop) // 2]
        while len(nextGen) < GeneralParameter.pop:
            
            prnt1, prnt2 = tournament_selection(pop, GeneralParameter.Tour)

            if GeneralParameter.mutation_type == 2:
                prnt1_sg = prnt1[-1]
                prnt2_sg = prnt2[-1]
                prnt1 = prnt1[:-1]
                prnt2 = prnt2[:-1]
            if GeneralParameter.mutation_type == 3:
                prnt1_sg = prnt1[len(prnt1) // 2:]
                prnt2_sg = prnt2[len(prnt2) // 2:]
                prnt1 = prnt1[:len(prnt1) // 2]
                prnt2 = prnt2[:len(prnt2) // 2]
            offs1, offs2 = crossover(prnt1, prnt2)
            if GeneralParameter.mutation_type == 2:
                offs1.append(prnt1_sg)
                offs2.append(prnt2_sg)
            if GeneralParameter.mutation_type == 3:
                offs1 += prnt1_sg
                offs2 += prnt2_sg
            mutated_offs1 = mutation(offs1)
            mutated_offs2 = mutation(offs2)
            a = mutated_offs1
            if GeneralParameter.mutation_type == 1:
                b = styblinsTouri_tang(a)
            elif GeneralParameter.mutation_type == 2:
                b = styblinsTouri_tang(a[:-1])
            elif GeneralParameter.mutation_type == 3:
                b = styblinsTouri_tang(a[:len(a) // 2])
            nextGen.append(new_solution(a, b))
            a = mutated_offs2
            if GeneralParameter.mutation_type == 1:
                b = styblinsTouri_tang(a)
            elif GeneralParameter.mutation_type == 2:
                b = styblinsTouri_tang(a[:-1])
            elif GeneralParameter.mutation_type == 3:
                b = styblinsTouri_tang(a[:len(a) // 2])
            nextGen.append(new_solution(a, b))
        pop = nextGen
        display(pop, gen_count)
        if gen_count >= GeneralParameter.maxgen:
            print("Stopped after,   " + str(gen_count) + " generations.")
            breaTour

BenchmarTour.run(main)

