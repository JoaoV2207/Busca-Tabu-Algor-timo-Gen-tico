
import numpy as np
import json
import time
# seeds = np.random.randint(1,1000, size = 5)
seeds = [153, 442, 929, 384, 324]


# # GA (Genetic Algorithm)

# In[7]:


def randomSolution(bases_count, limit):
  solution = np.zeros(bases_count, dtype=bool)
  indexes = rng.choice(list(range(bases_count)), limit, replace= False)
  solution[indexes] = 1
  return solution


# In[8]:


def mutation(solution):
  ones = np.where(solution == 1)[0]
  zeros = np.where(solution == 0)[0]
  one = rng.choice(ones, 1)
  zero = rng.choice(zeros, 1)
  solution = np.copy(solution)
  solution[one] = 0
  solution[zero] = 1
  return solution


# In[9]:


def crossover(solution1,solution2):
  child1 = solution1*solution2
  child2 = solution1*solution2

  uniques_solution = solution1 + solution2 - child1*2

  ones = np.where(uniques_solution == 1)[0]
  rng.shuffle(ones)
  halfones = len(ones)//2
  firsthalf = ones[:halfones]
  secondhalf = ones[halfones:]

  child1[firsthalf] = 1
  child2[secondhalf] = 1

  return (child1,child2)


# In[10]:


def tournament_selection(population_fitness, tournamentk, all_pop_ind, pop_size):
  selection = []
  for i in range(pop_size-2):
    individuals = rng.choice(all_pop_ind, size=tournamentk, replace=False)
    bestFitness = 0
    best = -1
    for i in range(tournamentk):
      this_fitness = population_fitness[individuals[i]]
      if(this_fitness > bestFitness):
        bestFitness = this_fitness
        best = individuals[i]
    selection.append(best)
  return selection


# In[11]:


def reproduction(population, pop_size, selection, pmut):
  newPopulation = []
  for i in range(pop_size//2-1):
    ind1 = population[selection[i*2]]
    ind2 = population[selection[i*2+1]]
    typeReproduction = rng.random()
    if typeReproduction < pmut:
      # mutacao
      newPopulation.append(mutation(ind1))
      newPopulation.append(mutation(ind2))
    else:
      # crossover
      (new1,new2) = crossover(ind1, ind2)
      newPopulation.append(new1)
      newPopulation.append(new2)
  return newPopulation


# In[12]:


def objectiveFunction(solution, coverMatrix, numPeopleHelpedList):
  coverage = coverMatrix[:,solution]
  coverage = np.logical_or.reduce(coverage, axis=1)
  solutionUtility = coverage*numPeopleHelpedList
  return np.sum(solutionUtility)


# In[13]:


def solveGeneticAlgorithm(pop_size, ngen, pmut, tournamentk, coverMatrix, numPeopleHelpedList, limit):
  allPop = list(range(pop_size))
  bases_count = coverMatrix.shape[1]

  population = [randomSolution(bases_count, limit) for i in range(pop_size)]
  for i in range(ngen):
    population_fitness = [objectiveFunction(ind, coverMatrix, numPeopleHelpedList) for ind in population]
    # if i % (ngen//4) == 0:
      # print(i, np.mean(population_fitness), np.max(population_fitness))
    indexBest = population_fitness.index(np.max(population_fitness))
    best = population[indexBest].copy()

    selection = tournament_selection(population_fitness, tournamentk, allPop, pop_size)
    newPopulation = reproduction(population, pop_size, selection, pmut)

    # elitism
    newPopulation.append(best)
    newPopulation.append(best)

    population = newPopulation

  population_fitness = [objectiveFunction(ind, coverMatrix, numPeopleHelpedList) for ind in population]
  print(i, np.mean(population_fitness), np.max(population_fitness))

  indexBest = population_fitness.index(max(population_fitness))
  best = population[indexBest].copy()
  return best


# In[15]:


resultDict = {}
f = open('/content/casos_teste2x.json')
data = json.load(f)
for instance in data.keys():
  data[instance]['matriz'] = np.asarray(data[instance]['matriz'])
  data[instance]['populacao'] = np.asarray(data[instance]['populacao'])


# In[14]:


data.keys()


# In[ ]:


matriz = data['kroA150']['matriz']
populacao = data['kroA150']['populacao']
p = int(np.ceil(matriz.shape[1]*0.2))



result = []
for seed in seeds:
  rng = np.random.default_rng(seed=seed)
  best = solveGeneticAlgorithm(400, 100, 0.7, 5, coverMatrix = matriz, numPeopleHelpedList = populacao, limit = p)
  result.append(objectiveFunction(best,matriz,populacao))
result = np.array(result)


# In[ ]:


resultDict = {}


# In[ ]:


for instance in data.keys():
  print(instance)
  matriz = data[instance]['matriz']
  populacao = data[instance]['populacao']
  p = int(np.ceil(matriz.shape[1]*0.2))

  result = []
  duration = []
  for seed in seeds:
    rng = np.random.default_rng(seed=seed)
    start_time = time.time()
    best = solveGeneticAlgorithm(400, 100, 0.8, 5, coverMatrix = matriz, numPeopleHelpedList = populacao, limit = p)
    end_time = time.time()
    duration.append(end_time-start_time)

    result.append(objectiveFunction(best,matriz,populacao))
  result = np.array(result)
  duration = np.array(duration)
  resultDict[instance] = {'mean': result.mean(), 'std': result.std(), 'duration':duration.mean()}


# In[ ]:


resultDict


# In[ ]:





# In[ ]:


for result in resultDict:
  print(resultDict[result]['duration'])


# In[ ]:


with open('resultGA2.json', 'w') as json_file:
    json.dump(resultDict, json_file)


# In[ ]:


for i in resultDict:
  print(i, ',', resultDict[i]['duration'],';')


# In[ ]:


resultDict


# 
