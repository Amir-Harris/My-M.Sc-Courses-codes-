import random
import math
import statistics
import time
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy
import  styblinsky_tang

best_per_gen = []
not_behbod = 0
class Tuner_Parameter:
    # function parameters

    # Tuner parameters
    initial_A = 10
    B = 2
    C = 100
    K = 5
    pv_num = initial_A
    pop_size_min_range = 4
    pop_size_max_range = 100

    # GENITOR_delete_worst_vector_percent = 20
    # nmu_plus_deleted_new_vector = -5

    mutation_rate_min_range = 0.1
    mutation_rate_max_range = 0.5

    crossover_rate_min_range = 0.5
    crossover_rate_max_range = 1

    # EA parameters
    maxGeneration = 100
    stall_iteration = 10

    # Crossover and Mutation Rates
    mutationrate = 0.3
    crossoverrate = 0.78


    selection_Probabilities_type = 'FPS'
    parent_selection_type = 'RW'
    K = 5  # tournament
    Linear_Ranking_tuner_s = 1.5  # s: 1 < s <= 2
    over_selection = False
    over_Selection_n_percent_best = 25

    selection_Probabilities_dict = {'FPS': 'Fitness-Proportionate Selection', 'LR': 'Linear Ranking',
                                    'ER': 'Exponential Ranking'}
    parent_selection_dict = {'RW': 'Roulette wheel Selection', 'SUS': 'stochastic universal sampling (SUS)',
                             'tournament': 'tournament selection', 'uniform': 'Uniform selection'}

    # Report parameters
    show_plot = True

bestFit_tun = None
timings_tests_per_vector = []
timings_per_gen = []

class new_p_v:
    def __init__(self, parameter_vector, utility):
        self.parameter_vector = parameter_vector
        self.utility = utility


def Mean_Best_Fitness(pv):
    global timings_tests_per_vector
    Best_Fitness = []
    MBF = None
    p_size = pv[0]
    mutation_rate = pv[1]
    crossover_rate = pv[2]
    for i in range(0, Tuner_Parameter.B):  # number of tests per vector
        startTime = time.time()
        bf = styblinsky_tang.main(p_size, mutation_rate, crossover_rate, Tuner_Parameter.C)
        seconds = time.time() - startTime
        timings_tests_per_vector.append(seconds)
        Best_Fitness.append(bf)
    sbf = sum(Best_Fitness)
    MBF = round(float(sbf) / len(Best_Fitness), 8)
    return MBF


# initialize parameter vectors values
def initialize_parameter_vectors():
    pv_ini = []

    while len(pv_ini) < Tuner_Parameter.pv_num:
        pv = []
        pop_size = random.randint(Tuner_Parameter.pop_size_min_range, Tuner_Parameter.pop_size_max_range)
        pv.append(pop_size)
        crossover_rate = random.uniform(Tuner_Parameter.crossover_rate_min_range, Tuner_Parameter.crossover_rate_max_range)
        pv.append(crossover_rate)
        mutation_rate = random.uniform(Tuner_Parameter.mutation_rate_min_range, Tuner_Parameter.mutation_rate_max_range)
        pv.append(mutation_rate)
        utility = Mean_Best_Fitness(pv)
        pv_ini.append(new_p_v(pv, utility))
    return pv_ini




def Fitness_Proportionate_tuner(pop):
    maxNum = max(c.utility for c in pop)
    minNum = min(c.utility for c in pop)
    if maxNum == minNum:
        maxNum = 0
    newFits = []

    for c in pop:
        newFits.append(maxNum + (-c.utility))
    sumFit = sum(c for c in newFits)
    p = []
    if sumFit == 0:
        sumFit == 1
    for i in newFits:
        p.append(i / sumFit)
    # sump = sum(c for c in p)
    return p


def Linear_Ranking_tuner(pop, s):
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


def RouletteWheelSelection_tuner_tuner(pop):
    c = len(pop) - 0.58
    p = []
    for idx in range(1, len(pop) + 1):
        pi = (1 - math.exp(-idx)) / float(c)
        p.append(pi)
    p.reverse()

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

# ----- uniform_Selection
def uniform_selection(pop):
    parents = []
    for i in range(2):
        parents.append(pop[random.randint(0, len(pop) - 1)])
    return parents[0].gens, parents[1].gens



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
    Round_robin_sel_num = Tuner_Parameter.Pop_size / 100 * sel_num_percent
    for i in range(Round_robin_sel_num):
        index = pop_wins[i].index
        survivors.append(pop[index])
    return survivors

# crossover
def crossover(pv1, pv2):

    offsp1 = copy.deepcopy(pv1)
    offsp2 = copy.deepcopy(pv2)
    for i in range(0, len(pv1)):
        if random.random() <= Tuner_Parameter.crossoverrate:
            temp = offsp1[i]
            offsp1[i] = offsp2[i]
            offsp2[i] = temp
    return offsp1, offsp2

# mutation
def mutation(pv):
    # uniform mutation
    mpv = copy.deepcopy(pv)
    if random.random() <= Tuner_Parameter.mutationrate:
            mpv[0] = random.randint(Tuner_Parameter.pop_size_min_range, Tuner_Parameter.pop_size_max_range)
    if random.random() <= Tuner_Parameter.mutationrate:
            mpv[1] = random.uniform(Tuner_Parameter.crossover_rate_min_range, Tuner_Parameter.crossover_rate_max_range)
    if random.random() <= Tuner_Parameter.mutationrate:
            mpv[2] = random.uniform(Tuner_Parameter.mutation_rate_min_range, Tuner_Parameter.mutation_rate_max_range)
    return mpv


def parents_selection_tuner(pvs):

    #parent Selection
    if (Tuner_Parameter.selection_Probabilities_type == 'FPS'):
        p = Fitness_Proportionate_tuner(pvs)
    if (Tuner_Parameter.selection_Probabilities_type == 'LR'):
        p = Linear_Ranking_tuner(pvs, Tuner_Parameter.Linear_Ranking_tuner_s)
    if (Tuner_Parameter.selection_Probabilities_type == 'ER'):
        p = RouletteWheelSelection_tuner_tuner(pvs)
    if (Tuner_Parameter.parent_selection_type == 'RW'):
        i1 = RouletteWheelSelection(p)
        i2 = RouletteWheelSelection(p)
        parent1 = pvs[i1].parameter_vector
        parent2 = pvs[i2].parameter_vector
    if (Tuner_Parameter.parent_selection_type == 'SUS'):
        i = stochastic_universal_sampling(p, 2)
        parent1 = pvs[i[0]].gens
        parent2 = pvs[i[1]].gens
    if (Tuner_Parameter.parent_selection_type == 'tournament'):
        parent1, parent2 = tournament_selection(pvs, Tuner_Parameter.K)
    if (Tuner_Parameter.parent_selection_type == 'uniform'):
        parent1, parent2 = uniform_selection(pvs)
    return parent1, parent2


def variation(parent1, parent2):

    offspring1, offspring2 = crossover(parent1, parent2)

    mutated_offspring1 = mutation(offspring1)
    mutated_offspring2 = mutation(offspring2)

    utility1 = Mean_Best_Fitness(mutated_offspring1)
    utility2 = Mean_Best_Fitness(mutated_offspring2)

    pv_offspring1 = new_p_v(mutated_offspring1, utility1)
    pv_offspring2 = new_p_v(mutated_offspring2, utility2)
    return pv_offspring1, pv_offspring2

def next_pvs_generation(pvs):
    pvs_len = None


    # delete_worst_num = int(math.ceil(len(pvs) * Tuner_Parameter.GENITOR_delete_worst_vector_percent / 100.0))
    pvs.sort(key=lambda x: (x.utility, x.parameter_vector[0]))
    all_utility = [];
    for v in pvs:
        all_utility.append(v.utility);
    mean_uitility =numpy.mean(all_utility);
    for idx, val in enumerate(pvs):
        if val.utility > mean_uitility:
            end_idx = idx;
            break;
    delete_worst_num = len(pvs) - end_idx;


    if len(pvs) - delete_worst_num <= 1:
        pvs_nex = copy.deepcopy(pvs[0])
        pvs_len = 1
    else:
        pvs_nex = copy.deepcopy(pvs[:len(pvs) - delete_worst_num])

    next_generation = copy.deepcopy(pvs_nex)

    if pvs_len == 1:
        next_generation.utility = Mean_Best_Fitness(next_generation.parameter_vector)
    else:
        for i in range(0, len(next_generation)):
            next_generation[i].utility = Mean_Best_Fitness(next_generation[i].parameter_vector)

        Add_new_vector_num = delete_worst_num / 4
        # Add_new_vector_pair_num = int(math.ceil(Add_new_vector_num / 2.0))
        for _ in range(0, Add_new_vector_num):

            p1, p2 = parents_selection_tuner(pvs_nex)
            pv1, pv2 = variation(p1, p2)
            next_generation.append(pv1)
            # if len(next_generation) < Tuner_Parameter.initial_A:
            next_generation.append(pv2);

    return next_generation, delete_worst_num

def display_tuner(pvs, gen_count, replaced_num):
    global bestFit_tun
    global not_behbod
    global timings_tests_per_vector
    if isinstance(pvs, list):
        sum_utility = sum((c.utility for c in pvs))
        avr_utility = float(sum_utility) / len(pvs)
        pvs.sort(key=lambda x: (x.utility, x.parameter_vector[0]))
        best_utility = copy.deepcopy(pvs[0])
        worst_utility = copy.deepcopy(pvs[-1])
        len_pvs = len(pvs)
    else:
        len_pvs = 1
        avr_utility = pvs.utility
        best_utility = copy.deepcopy(pvs)
        worst_utility = copy.deepcopy(pvs)

    behbod = ""
    not_behbod += 1
    if bestFit_tun == None:
        bestFit_tun = copy.deepcopy(best_utility)
    if round(bestFit_tun.utility, 8) > round(best_utility.utility, 8) or \
            (round(bestFit_tun.utility, 8) == round(best_utility.utility, 8) and bestFit_tun.parameter_vector[0] > best_utility.parameter_vector[0]):
        bestFit_tun = copy.deepcopy(best_utility)
        behbod = "*"
        not_behbod = 0
    best_per_gen.append(best_utility)
    print(
    "####in Generation({}),    Best utility={},  Worst utility={},  Average utility={},   number of vectors tested={} (-,+ {}),    {}"
    .format(gen_count, best_utility.utility, worst_utility.utility, avr_utility, len_pvs, replaced_num,
            behbod))
    print("Best parameters: Population size = {},    Crossover rate={},     Mutation rate={}".
          format(best_utility.parameter_vector[0], best_utility.parameter_vector[1], best_utility.parameter_vector[2], ))

    mean_time_tests_per_vector = statistics.mean(timings_tests_per_vector)
    sum_time_tests_per_vector = sum(timings_tests_per_vector)
    print("For {} number of tests per {} vector, Run time(sec): mean={:3.2f}  stdev={:3.2f}  sum={:3.2f}"
          .format(Tuner_Parameter.B, len_pvs, mean_time_tests_per_vector,
                  statistics.stdev(timings_tests_per_vector, mean_time_tests_per_vector), sum_time_tests_per_vector))
    timings_tests_per_vector = []
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
            print("{} {:3.2f} {:3.2f}".format(1 + i, mean, statistics.stdev(timings, mean) if i > 1 else 0))

def main_func():
    global timings_per_gen
    global not_behbod
    print('Algorithm START with this parameters:')
    print('     Initial number of vectors tested(A) = {}'.format(Tuner_Parameter.initial_A))
    print('     number of tests per vector(B) = {}'.format(Tuner_Parameter.B))
    print('     number of fitness evaluations per test(C) = {}'.format(Tuner_Parameter.C))
    print('     Crossover type = Uniform,    Crossover rate = {}'.format(Tuner_Parameter.crossoverrate))
    print('     Mutation type = Uniform,    Mutation rate = {}'.format(Tuner_Parameter.mutationrate))
    print('     parent selection Probabilities = {},    Implementing Selection Probabilities = {}'
          .format(Tuner_Parameter.selection_Probabilities_dict[Tuner_Parameter.selection_Probabilities_type],
                  Tuner_Parameter.parent_selection_dict[Tuner_Parameter.parent_selection_type], ))
    print('     Termination: Max Generation {} Reached or not improvment for {} Generation or P.V = 1'
          .format(Tuner_Parameter.maxGeneration, Tuner_Parameter.stall_iteration))


    start_gen_Time = time.time()
    parameter_vectors = initialize_parameter_vectors()
    gen_count = 1
    seconds = time.time() - start_gen_Time
    timings_per_gen.append(seconds)
    display_tuner(parameter_vectors, gen_count, 0)
    while True:
        gen_count += 1
        start_gen_Time = time.time()
        parameter_vectors, replaced_num = next_pvs_generation(parameter_vectors)
        seconds = time.time() - start_gen_Time
        timings_per_gen.append(seconds)
        # when len(parameter_vectors) == 1
        if not isinstance(parameter_vectors, list):
            len_parameter_vectors = 1
        else:
            len_parameter_vectors = len(parameter_vectors)
        display_tuner(parameter_vectors, gen_count, replaced_num)
        if gen_count >= Tuner_Parameter.maxGeneration or not_behbod >= 10 or not isinstance(parameter_vectors, list):
            print("#### Stopped after,   " + str(gen_count) + " generations.")
            print('##### Best for all Generation =')
            print('##### best utility={} for Test all A={} vectors for B={} in C={} Generation.'.format(
                    bestFit_tun.utility, len_parameter_vectors, Tuner_Parameter.B, Tuner_Parameter.C))
            print("Population size = {},    Crossover rate={},     Mutation rate={}".
                  format(bestFit_tun.parameter_vector[0], bestFit_tun.parameter_vector[1], bestFit_tun.parameter_vector[2],))
            mean_time_per_gen = statistics.mean(timings_per_gen)
            sum_time_per_gen = sum(timings_per_gen)
            print("For (B)={} number of tests *(C)={}, per (A)<={} vector, Run time(sec): mean={:3.2f}  stdev={:3.2f}  sum={:3.2f}"
                  .format(Tuner_Parameter.B, Tuner_Parameter.C, len_parameter_vectors, mean_time_per_gen,
                          statistics.stdev(timings_per_gen, mean_time_per_gen),
                          sum_time_per_gen))
            bpg = []
            for p in best_per_gen:
                bpg.append(p.utility)
            if Tuner_Parameter.show_plot:
                plt.plot(bpg)
                plt.title("Best Utility per Generations")
                plt.xlabel('Generations')
                plt.ylabel('Utility')
                plt.show()

                plt.plot(timings_per_gen)
                plt.title("Run Time per generation")
                plt.xlabel('Generations')
                plt.ylabel('Run Time')
                plt.show()

            break


Benchmark.run(main_func)

