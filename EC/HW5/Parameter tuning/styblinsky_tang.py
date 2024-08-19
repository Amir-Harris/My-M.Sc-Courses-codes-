import random
import math
import statistics
import time
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy

best_per_generation = []

class GeneralParameter:

    dimensions = 5
    minrange = -5
    maxrange = 5
    graph_disp = False
    show_plot = False
    Pop_size = 30
    maxGeneration = 100


    crossover_type = 3
    crossover_type_dict = {1: 'Single arithmetic crossover', 2: 'Simple arithmetic crossover',
                           3: 'Whole arithmetic crossover'}

    mutation_type = 1
    mutation_type_dict = {1: 'Uniform Mutation', 2: 'Uncorrelated mutation with one sigma',
                          3: 'Uncorrelated mutation with n sigma'}
    # Crossover and Mutation Rates
    mutationrate = 0.3
    crossoverrate = 0.78

    tao = 1 / math.sqrt(2 * math.sqrt(2))
    taop = 1 / math.sqrt(4)
    eplsz = 0.01
    alpha = 0.5
    selection_Probabilities_type = 'LR'
    parent_selection_type = 'tournament'
    K = 5  # tournament size
    linear_Ranking_s = 1.5  # s: 1 < s <= 2
    over_selection = False
    over_Selection_n_percent_best = 25

    selection_Probabilities_dict = {'FPS': 'Fitness-Proportionate Selection', 'LR': 'Linear Ranking',
                                    'ER': 'Exponential Ranking'}
    parent_selection_dict = {'RW': 'Roulette wheel Selection', 'SUS': 'stochastic universal sampling (SUS)',
                             'tournament': 'tournament selection', 'uniform': 'Uniform selection',
                             'DE': 'Differential Evolution'}

    Survivor_selection_types = 'El'
    Survivor_selection_dict = {'El': 'Elitism', 'GE': 'GENITOR: a.k.a. "delete-worst',
                               'Rr': 'Round-robin tournament',
                               'DE': 'DE: copmare each parent with its child (Select all parents for mutation and next crossover) '}
    Elites_percent = 35
    u_landa_type = 'u+landa'
    landa = 3
    Round_robin_tournament_q = 10
    Round_robin_tournament_percent = 50
    diplay_results = False
    reportShowLevel = 2
    delayshow_point = 1
    delayshow = 2


bestFit = None

class new_solution:
    def __init__(self, gens, FitnessValue):
        self.gens = gens
        self.FitnessValue = FitnessValue
        self.age = 1


def styblinski_tang(x):
    dSum = 0.0
    for c in x:
        dSum += (c ** 4 - 16 * c ** 2 + 5 * c)
    return 0.5 * dSum


def initialize_population():
    pop_ini = []
    if (GeneralParameter.mutation_type == 1 or GeneralParameter.mutation_type == 4):
        gen_len = GeneralParameter.dimensions
    if (GeneralParameter.mutation_type == 2):
        gen_len = GeneralParameter.dimensions + 1
    if (GeneralParameter.mutation_type == 3):
        gen_len = GeneralParameter.dimensions * 2
    while len(pop_ini) < GeneralParameter.Pop_size:
        a = [random.uniform(GeneralParameter.minrange, GeneralParameter.maxrange) for _ in range(gen_len)]
        if GeneralParameter.mutation_type == 1 or GeneralParameter.mutation_type == 4:
            b = styblinski_tang(a)
        elif GeneralParameter.mutation_type == 2:
            b = styblinski_tang(a[:-1])
        elif GeneralParameter.mutation_type == 3:
            b = styblinski_tang(a[:len(a) // 2])
        pop_ini.append(new_solution(a, b))
    return pop_ini



def Fitness_Proportionate(pop):
    maxNum = max(c.FitnessValue for c in pop)
    minNum = min(c.FitnessValue for c in pop)
    if maxNum == minNum:
        maxNum = 0
    newFits = []
    # if maxNum > 0:
    #     singn
    for c in pop:
        newFits.append(maxNum + (-c.FitnessValue))
    sumFit = sum(c for c in newFits)
    p = []
    if sumFit == 0:
        sumFit == 1
    for i in newFits:
        p.append(i / sumFit)
    # sump = sum(c for c in p)
    return p

def Linear_Ranking(pop, s):
    pop.sort(key=lambda x: x.FitnessValue)
    pop.reverse()
    u = len(pop)
    p = []
    for idx in range(0, u):
        pi = (2 - s) / u + (2 * idx * (s - 1)) / (u * (u - 1))
        p.append(pi)
    # sump = sum(c for c in p)
    p.reverse()
    return p


def Exponential_Ranking(pop):
    c = len(pop) - 0.58
    p = []
    for idx in range(1, len(pop) + 1):
        pi = (1 - math.exp(-idx)) / float(c)
        p.append(pi)
    p.reverse()
    # correct p ?!!!!
    sump = sum(c for c in p)
    bagi = (1.0 - sump)
    n = (len(pop))
    # sahm =  bagi / n
    p[0] += bagi
    # for idx in range(0, c ):
    #     p[idx] += bagi
    sump2 = sum(c for c in p)
    # p[0] += 1 - sump2
    # sump3 = sum(c for c in p)
    return p


def RouletteWheelSelection(p):
    r = random.uniform(0, 1)
    # c = itertools.accumulate(p)
    c = [0] * len(p)
    c[0] = p[0]
    for i in range(1, len(p)):
        c[i] = c[i - 1] + p[i]
    # j=0
    for i in range(len(c)):
        if r <= c[i]:
            break
            # j+=1
    return i


def stochastic_universal_sampling(p, landa):
    # c = itertools.accumulate(p)
    a = [0] * len(p)
    a[0] = p[0]
    for i in range(1, len(p)):
        a[i] = a[i - 1] + p[i]
    mating_pool = []
    current_member = i = 0
    r = random.uniform(0, 1.0 / landa)
    while current_member < landa:
        while r <= a[i]:
            # mating_pool[current_member] = i
            mating_pool.append(i)
            r = r + 1.0 / landa
            # current_member = current_member + 1
            current_member = len(mating_pool)
        i = i + 1
    return mating_pool


def tournament_selection(pop, K):
    contestant = []
    for i in range(K):
        contestant.append(pop[random.randint(0, len(pop) - 1)])
    contestant.sort(key=lambda x: x.FitnessValue)
    return contestant[0].gens, contestant[1].gens


def uniform_selection(pop):
    parents = []
    for i in range(2):
        parents.append(pop[random.randint(0, len(pop) - 1)])
    return parents[0].gens, parents[1].gens


"""
    Survivor Selection parameters
            'Rr' = Round-robin tournament
"""


class pop_win_new:
    def __init__(self, index):
        self.index = index
        self.wins = 0


def Round_robin_tournament(pop, q, sel_num_percent):
    pop_wins = []
    survivors = []
    for i in range(len(pop)):
        pop_wins.append(pop_win_new(i))
    for i in range(len(pop)):
        for j in range(q):
            if pop[i].FitnessValue < pop[random.randint(0, len(pop) - 1)].FitnessValue:
                pop_wins[i].wins += 1
    pop_wins.sort(key=lambda x: x.wins)
    pop_wins.reverse()
    Round_robin_sel_num = GeneralParameter.Pop_size / 100 * sel_num_percent
    for i in range(Round_robin_sel_num):
        index = pop_wins[i].index
        survivors.append(pop[index])
    return survivors



def mutation(parent):
    offspring = parent
    if (GeneralParameter.mutation_type == 1):  # Uniform Mutation
        for i in range(len(parent)):
            if (random.random() <= GeneralParameter.mutationrate):
                offspring[i] = random.uniform(GeneralParameter.minrange, GeneralParameter.maxrange)
    if (GeneralParameter.mutation_type == 2):  # Uncorrelated mutation with one sigma
        randnorm = random.random()
        parent[-1] = parent[-1] * math.exp(GeneralParameter.tao * randnorm)
        # if parent[-1] < GeneralParameter.minrange / 100:
        #     parent[-1] = -0.01
        # if parent[-1] > GeneralParameter.maxrange / 100:
        #     parent[-1] = 0.01
        if (parent[-1] < GeneralParameter.eplsz):
            parent[-1] = GeneralParameter.eplsz
        for i in range(0, len(parent) - 1):
            parent[i] = parent[i] + parent[-1] * random.random()
    if (GeneralParameter.mutation_type == 3):  # Uncorrelated mutation with n sigma
        randnorm = random.random()
        for i in range(len(parent) // 2, len(parent)):
            parent[i] = parent[i] * math.exp(GeneralParameter.taop * randnorm + GeneralParameter.tao * random.random())
            if parent[-1] < -0.01:
                parent[-1] = -0.01
            if parent[-1] > 0.01:
                parent[-1] = 0.01
            if parent[i] < GeneralParameter.eplsz:
                parent[i] = GeneralParameter.eplsz
        for i in range(0, len(parent) // 2):
            if not (parent[i] >= GeneralParameter.minrange and parent[i] <= GeneralParameter.maxrange):
                parent[i] = parent[i] + parent[i + len(parent) // 2] * random.random()
    return offspring


def crossover(parent1, parent2):
    offsp1 = []
    offsp2 = []
    if (GeneralParameter.crossover_type == 1):  # Single arithmetic crossover
        if (random.random() <= GeneralParameter.crossoverrate):
            crsPoint = random.randint(0, len(parent1) - 1)
            offsp1 = copy.deepcopy(parent1)
            offsp2 = copy.deepcopy(parent2)
            offsp1[crsPoint] = parent1[crsPoint] * GeneralParameter.alpha + (1 - GeneralParameter.alpha) * parent2[
                crsPoint]
            offsp2[crsPoint] = parent2[crsPoint] * GeneralParameter.alpha + (1 - GeneralParameter.alpha) * parent1[
                crsPoint]
        else:
            offsp1 = copy.deepcopy(parent1)
            offsp2 = copy.deepcopy(parent2)

    if (GeneralParameter.crossover_type == 2):  # Simple arithmetic crossover
        if (random.random() <= GeneralParameter.crossoverrate):
            crsPoint = random.randint(0, len(parent1) - 1)
            offsp1 = copy.deepcopy(parent1)
            offsp2 = copy.deepcopy(parent2)
            for p in range(crsPoint, len(parent1) - 1):
                offsp1[p] = parent1[p] * GeneralParameter.alpha + (1 - GeneralParameter.alpha) * parent2[p]
                offsp2[p] = parent2[p] * GeneralParameter.alpha + (1 - GeneralParameter.alpha) * parent1[p]
        else:
            offsp1 = copy.deepcopy(parent1)
            offsp2 = copy.deepcopy(parent2)

    if (GeneralParameter.crossover_type == 3):  # Whole arithmetic crossover
        if (random.random() <= GeneralParameter.crossoverrate):
            offsp1 = copy.deepcopy(parent1)
            offsp2 = copy.deepcopy(parent2)
            for p in range(0, len(parent1) - 1):
                offsp1[p] = parent1[p] * GeneralParameter.alpha + (1 - GeneralParameter.alpha) * parent2[p]
                offsp2[p] = parent2[p] * GeneralParameter.alpha + (1 - GeneralParameter.alpha) * parent1[p]
        else:
            offsp1 = copy.deepcopy(parent1)
            offsp2 = copy.deepcopy(parent2)

    return offsp1, offsp2


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
                ott.append(styblinski_tang([x[0][i], x[1][j]]))
            o.append(ott)
            xt.append(xtt)
            yt.append(ytt)
        px = numpy.array(xt)
        py = numpy.array(yt)
        op = numpy.array(o)
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_wireframe(px, py, op, rstride=10, cstride=10)  # cmap=cm.coolwarm,linewidth=0, antialiased=False)
        # plt.show(block=0)
        # time.sleep(10)
        # plt.close()
        return ax

    else:
        for c in x:
            o.append(styblinski_tang(c))
        plt.figure(1)
        plt.ylabel('function ')
        plt.plot(x, o)
        plt.show(block=0)
        time.sleep(2)
        plt.close()


def display(pop, gen_count):
    global bestFit
    sFit = sum((c.FitnessValue for c in pop))
    sAge = sum((c.age for c in pop))
    avrFit = sFit / len(pop)
    avrAge = float(sAge) / len(pop)
    # maxFit = max((c.FitnessValue for c in pop))
    # minFit = min((c.FitnessValue for c in pop))
    maxFit = None
    minFit = None
    maxAge = None
    minAge = None
    overall_maxAge = None
    overall_minAge = None
    for c in pop:
        if maxFit == None or maxFit < c.FitnessValue:
            maxFit = c.FitnessValue
            maxAge = c.age
            if overall_maxAge == None or overall_maxAge < c.age:
                overall_maxAge = c.age
        if minFit == None or minFit > c.FitnessValue:
            minFit = c.FitnessValue
            minAge = c.age
            if overall_minAge == None or overall_minAge > c.age:
                overall_minAge = c.age
        # if overal_maxAge is not maxFit show as float (and so on min...)
        if overall_maxAge == None or overall_maxAge < c.age:
            overall_maxAge = float(c.age)
        if overall_minAge == None or overall_minAge > c.age:
            overall_minAge = float(c.age)
    behbod = ""
    if bestFit == None:
        bestFit = minFit
    elif bestFit > minFit:
        GeneralParameter.bestFit = minFit
        behbod = "*"
    best_per_generation.append(minFit)
    if GeneralParameter.diplay_results:
        print(
            "***in Generation({}) [Fitness & Age, Overall_Age]= average({},{}) max({} & {}, {}) min({} & {}, {}) Ind Numb({})    {}"
                .format(gen_count, avrFit, avrAge, maxFit, maxAge, overall_maxAge, minFit, minAge, overall_minAge, len(pop),behbod))

    if GeneralParameter.dimensions == 2 and GeneralParameter.graph_disp:
        a = []
        a.append(numpy.arange(-5, 5, 0.25))
        a.append(numpy.arange(-5, 5, 0.25))
        ax = display_p(a)
        time.sleep(GeneralParameter.delayshow)
        for c in pop:
            if GeneralParameter.mutation_type == 1 or GeneralParameter.mutation_type == 4:
                b = c.gens
            elif GeneralParameter.mutation_type == 2:
                b = c.gens[:-1]
            elif GeneralParameter.mutation_type == 3:
                b = c.gens[:len(c.gens) // 2]
            ax.scatter(b[0], b[1], styblinski_tang(b), c='r')  # , cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.title('in generation =' + str(gen_count) + '    ,     pop size =' + str(len(pop)))
        plt.pause(0.001)
        # time.sleep(GeneralParameter.delayshow)
        # plt.close()

    # return bestFit


class Benchmark:
    @staticmethod
    def run(function):
        timings = []
        # stdout = sys.stdout
        for i in range(1):
            # sys.stdout = None
            startTime = time.time()
            function()
            seconds = time.time() - startTime
            # sys.stdout = stdout
            timings.append(seconds)
            mean = statistics.mean(timings)
            if i < 10 or i % 10 == 9:
                if GeneralParameter.reportShowLevel < 3 and GeneralParameter.diplay_results:
                    print("{} {:3.2f} {:3.2f}".format(1 + i, mean, statistics.stdev(timings, mean) if i > 1 else 0))


def parents_selection(pop):
    parents = []
    # --- parent Selection --------------------------------------------------------------------------------------------
    if (GeneralParameter.selection_Probabilities_type == 'FPS'):
        p = Fitness_Proportionate(pop)
    if (GeneralParameter.selection_Probabilities_type == 'LR'):
        p = Linear_Ranking(pop, GeneralParameter.linear_Ranking_s)
    if (GeneralParameter.selection_Probabilities_type == 'ER'):
        p = Exponential_Ranking(pop)
    if (GeneralParameter.parent_selection_type == 'RW'):
        i1 = RouletteWheelSelection(p)
        i2 = RouletteWheelSelection(p)
        parent1 = pop[i1].gens
        parent2 = pop[i2].gens
    if (GeneralParameter.parent_selection_type == 'SUS'):
        i = stochastic_universal_sampling(p, 2)
        parent1 = pop[i[0]].gens
        parent2 = pop[i[1]].gens
    if (GeneralParameter.parent_selection_type == 'tournament'):
        parent1, parent2 = tournament_selection(pop, GeneralParameter.K)
    if (GeneralParameter.parent_selection_type == 'uniform'):
        parent1, parent2 = uniform_selection(pop)
    # if (GeneralParameter.parent_selection_type == 'OS'):
    #     parent1, parent2 = over_selection(pop, GeneralParameter.over_Selection_n_percent_best)
    # --- Crossover ---------------------------
    if GeneralParameter.mutation_type == 2:
        parent1_sigma = parent1[-1]
        parent2_sigma = parent2[-1]
        parent1 = parent1[:-1]
        parent2 = parent2[:-1]
    if GeneralParameter.mutation_type == 3:
        parent1_sigma = parent1[len(parent1) // 2:]
        parent2_sigma = parent2[len(parent2) // 2:]
        parent1 = parent1[:len(parent1) // 2]
        parent2 = parent2[:len(parent2) // 2]
    offspring1, offspring2 = crossover(parent1, parent2)
    if GeneralParameter.mutation_type == 2:
        offspring1.append(parent1_sigma)
        offspring2.append(parent2_sigma)
    if GeneralParameter.mutation_type == 3:
        offspring1 += parent1_sigma
        offspring2 += parent2_sigma
    # --- Mutation -----------------------------
    mutated_offspring1 = mutation(offspring1)
    mutated_offspring2 = mutation(offspring2)
    # --- add mutated_offspring1 ---------------------------------------
    a = mutated_offspring1
    if GeneralParameter.mutation_type == 1 or GeneralParameter.mutation_type == 4:
        b = styblinski_tang(a)
    elif GeneralParameter.mutation_type == 2:
        b = styblinski_tang(a[:-1])
    elif GeneralParameter.mutation_type == 3:
        b = styblinski_tang(a[:len(a) // 2])
    parents.append(new_solution(a, b))
    # --- add mutated_offspring2 ---------------------------------------
    a = mutated_offspring2
    if GeneralParameter.mutation_type == 1:
        b = styblinski_tang(a)
    elif GeneralParameter.mutation_type == 2:
        b = styblinski_tang(a[:-1])
    elif GeneralParameter.mutation_type == 3:
        b = styblinski_tang(a[:len(a) // 2])
    parents.append(new_solution(a, b))
    return parents


def variation_Survivor_Selection(pop, Survivor_sel_type, u_landa_type, gen_count):
    next_generation = []
    mating_pool = []
    # ------ surv Selection
    # --------------------------- 'El' = Elitism
    if GeneralParameter.Survivor_selection_types == 'El':
        Elites_num = GeneralParameter.Pop_size / 100 * GeneralParameter.Elites_percent
        pop.sort(key=lambda x: x.FitnessValue)
        # next_generation .reverse()
        next_generation = copy.deepcopy(pop[:Elites_num])
        # ---- Incress Age
        for i in range(0, len(next_generation)):
            next_generation[i].age += 1

    # --------------------------- 'Rr' = Round-robin tournament
    if GeneralParameter.Survivor_selection_types == 'Rr':
        next_generation = Round_robin_tournament(pop, GeneralParameter.Round_robin_tournament_q,
                                                 GeneralParameter.Round_robin_tournament_percent)
        for i in range(0, len(next_generation)):
            next_generation[i].age += 1

    # -------- Fill mating pool
    """
        u_landa type =
            'u,landa'
            'u+landa'
    """
    if GeneralParameter.u_landa_type == None:
        mating_pool = pop
    if GeneralParameter.u_landa_type == 'u,landa':
        next_generation = []
        mating_pool = []
        while len(mating_pool) < GeneralParameter.landa * GeneralParameter.Pop_size:
            mating_pool.extend(parents_selection(pop))

    if GeneralParameter.u_landa_type == 'u+landa':
        mating_pool = copy.deepcopy(pop)
        while len(mating_pool) < GeneralParameter.landa * GeneralParameter.Pop_size:
            mating_pool.extend(parents_selection(pop))

    if GeneralParameter.over_selection == True:
        num_best_sel_part = (
                                GeneralParameter.Pop_size / 100 * GeneralParameter.over_Selection_n_percent_best) + len(
            next_generation)
    else:
        num_best_sel_part = GeneralParameter.Pop_size
    mating_pool.sort(key=lambda x: x.FitnessValue)
    while len(next_generation) < GeneralParameter.Pop_size * 0.5:
        next_generation.extend(parents_selection(mating_pool[:num_best_sel_part]))

    while len(next_generation) < GeneralParameter.Pop_size:
        next_generation.extend(parents_selection(mating_pool[num_best_sel_part:]))

    return next_generation


def main(pop_size, cross_over_rate, mutation_rate, num_fitness_evaluation):
    global bestFit
    bestFit = None
    GeneralParameter.Pop_size = pop_size
    GeneralParameter.crossoverrate = cross_over_rate
    GeneralParameter.mutationrate = mutation_rate
    GeneralParameter.maxGeneration = num_fitness_evaluation
    if GeneralParameter.diplay_results:
        print('Algorithm START with this parameters:')
        print('     Population Size = {}'.format(GeneralParameter.Pop_size))
        print('     Crossover type = {},    Crossover rate = {}'.format(
            GeneralParameter.crossover_type_dict[GeneralParameter.crossover_type],
            GeneralParameter.crossoverrate))
        print('     Mutation type = {},    Mutation rate = {}'.format(
            GeneralParameter.mutation_type_dict[GeneralParameter.mutation_type],
            GeneralParameter.mutationrate))
        print('     parent selection Probabilities = {},    Implementing Selection Probabilities = {}'
              .format(GeneralParameter.selection_Probabilities_dict[GeneralParameter.selection_Probabilities_type],
                      GeneralParameter.parent_selection_dict[GeneralParameter.parent_selection_type], ))
        print('     survivor selection = {},    (u,landa) or (u+landa) = {}'.format(
            GeneralParameter.Survivor_selection_dict[GeneralParameter.Survivor_selection_types],
            GeneralParameter.u_landa_type))
        print('     Termination: Max Generation Reached,  Max Generation = {}'.format(GeneralParameter.maxGeneration))
    pop = initialize_population()
    gen_count = 1  # the generation counter

    display(pop, gen_count)
    while True:
        gen_count += 1
        pop = variation_Survivor_Selection(pop, GeneralParameter.parent_selection_type, GeneralParameter.u_landa_type,
                                           gen_count)
        display(pop, gen_count)
        # Stop if max Generation reached
        if gen_count >= GeneralParameter.maxGeneration:
            if GeneralParameter.diplay_results:
                print("Stopped after,   " + str(gen_count) + " generations.")
                print('Best fit per Generation:')
                print best_per_generation
            if GeneralParameter.show_plot:
                plt.plot(best_per_generation)
                plt.show()
            # for i in best_per_generation:
            #     print(i,',')
            break
    if GeneralParameter.diplay_results:
        print(bestFit)
    return bestFit

main(GeneralParameter.Pop_size, GeneralParameter.crossoverrate, GeneralParameter.mutationrate, 10)

