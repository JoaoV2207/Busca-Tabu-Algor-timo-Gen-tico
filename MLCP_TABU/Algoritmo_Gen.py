import numpy as np

def randomSolution(bases_count, limit, rng):
    solution = np.zeros(bases_count, dtype=bool)
    indexes = rng.choice(list(range(bases_count)), limit, replace=False)
    solution[indexes] = 1
    return solution

def mutation(solution, rng):
    ones = np.where(solution == 1)[0]
    zeros = np.where(solution == 0)[0]
    one = rng.choice(ones, 1)
    zero = rng.choice(zeros, 1)
    solution = np.copy(solution)
    solution[one] = 0
    solution[zero] = 1
    return solution

def crossover(solution1, solution2, rng):
    child1 = solution1 * solution2
    child2 = solution1 * solution2

    uniques_solution = solution1 + solution2 - child1 * 2

    ones = np.where(uniques_solution == 1)[0]
    rng.shuffle(ones)
    halfones = len(ones) // 2
    firsthalf = ones[:halfones]
    secondhalf = ones[halfones:]

    child1[firsthalf] = 1
    child2[secondhalf] = 1

    return (child1, child2)

def tournament_selection(population_fitness, tournamentk, all_pop_ind, pop_size, rng):
    selection = []
    for _ in range(pop_size - 2):
        individuals = rng.choice(all_pop_ind, size=tournamentk, replace=False)
        bestFitness = 0
        best = -1
        for i in range(tournamentk):
            this_fitness = population_fitness[individuals[i]]
            if this_fitness > bestFitness:
                bestFitness = this_fitness
                best = individuals[i]
        selection.append(best)
    return selection

def reproduction(population, pop_size, selection, pmut, rng):
    newPopulation = []
    for i in range(pop_size // 2 - 1):
        ind1 = population[selection[i * 2]]
        ind2 = population[selection[i * 2 + 1]]
        typeReproduction = rng.random()
        if typeReproduction < pmut:
            # Mutação
            newPopulation.append(mutation(ind1, rng))
            newPopulation.append(mutation(ind2, rng))
        else:
            # Crossover
            (new1, new2) = crossover(ind1, ind2, rng)
            newPopulation.append(new1)
            newPopulation.append(new2)
    return newPopulation

def objectiveFunction(solution, coverMatrix, numPeopleHelpedList):
    coverage = coverMatrix[:, solution]
    coverage = np.logical_or.reduce(coverage, axis=1)
    solutionUtility = coverage * numPeopleHelpedList
    return np.sum(solutionUtility)

def solveGeneticAlgorithm(pop_size, ngen, pmut, tournamentk, coverMatrix, numPeopleHelpedList, limit, rng):
    allPop = list(range(pop_size))
    bases_count = coverMatrix.shape[1]

    population = [randomSolution(bases_count, limit, rng) for _ in range(pop_size)]
    for i in range(ngen):
        population_fitness = [objectiveFunction(ind, coverMatrix, numPeopleHelpedList) for ind in population]
        indexBest = population_fitness.index(np.max(population_fitness))
        best = population[indexBest].copy()

        selection = tournament_selection(population_fitness, tournamentk, allPop, pop_size, rng)
        newPopulation = reproduction(population, pop_size, selection, pmut, rng)

        # Elitismo
        newPopulation.append(best)
        newPopulation.append(best)

        population = newPopulation

    population_fitness = [objectiveFunction(ind, coverMatrix, numPeopleHelpedList) for ind in population]
    indexBest = population_fitness.index(max(population_fitness))
    best = population[indexBest].copy()
    return best