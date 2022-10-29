import matplotlib.pyplot as plt
import cv2
import numpy as np



grayscale1 = cv2.imread("boothiGray.jpg")
grayscale2 = cv2.imread("groupGray.jpg")
groupImage = cv2.cvtColor(grayscale2, cv2.COLOR_BGR2GRAY)
boothi = cv2.cvtColor(grayscale1, cv2.COLOR_BGR2GRAY)
[groupColumns, groupRows] = np.shape(groupImage)
[boothiColumns, boothiRows] = np.shape(boothi)



# Initilization of random population
def populationInitilization(groupRows, groupColumns, population):
    randomPopulation = []
    for i in range(population):
        randomPopulation.append((np.random.randint(groupColumns), np.random.randint(groupRows)))

    return randomPopulation


# Assigning Fitness value (correlation) to every individual
def fitnessEvoluation(currentGeneration, groupImage, boothi):
    frame = []
    fitness_value = []
    fitness_values = {}
    for (x,y) in currentGeneration:
            frame.append(groupImage[x:x+boothiColumns, y:y+boothiRows])
    for i in range(population):
        if frame[i].shape == boothi.shape:
            frame[i] = frame[i] - frame[i].mean()
            boothi = boothi - boothi.mean()
            frame[i] = frame[i] / frame[i].std()
            boothi = boothi / boothi.std()
            value = np.mean(frame[i]*boothi)
            fitness_value.append(value)
        else:
            fitness_value.append(0)     
    for j in range(population):
        fitness_values[currentGeneration[j]] = fitness_value[j]

    return [fitness_values, fitness_value, frame]


# selection of best fit individual by sorting technique
def selection(currentPopulation, fitness_values):
    sort = []
    for i in fitness_values.values():
        sort.append(i)
    return sorted(sort)


# if threshold gets satisfied terminate else create new population
def faceDetection(currentGeneration, fitnessVal, sorted_fitness):
    newlyCreatedGeneration = newGeneration(currentGeneration, fitnessVal[1], sorted_fitness)
    key = []
    for i in fitnessVal[0].values():
        if (i >= fitnessThreshold):
            key.append([j for j in fitnessVal[0] if fitnessVal[0][j]== i])
            return [key, [], i]
        else:
            return newlyCreatedGeneration

    
# sorting population table accordingly
def newGeneration(currentGeneration, fitnessVal, sorted_fitness):
    newcreatedGeneration = []
    for i in range(population):
        for j in range(population):
            if (sorted_fitness[i]==fitnessVal[j]):
                fitnessVal[j] = -100
                newcreatedGeneration.append(currentGeneration[j])

    return newcreatedGeneration

    
# cross-over of Bits to do some diversity in generation
def crossOverOfBits(currentGeneration):
    binary_number = []
    out_array = np.asarray(currentGeneration, dtype=np.int)
    for i in out_array:
        x = np.binary_repr(i[0], width= 11)
        y = np.binary_repr(i[1], width= 11)
        binary_number.append((x,y))

    concat_binary = [''.join(i) for i in binary_number]
    result_list = []
    
    for i in range(0, population, 2):
        parent1_part = concat_binary[i]
        parent2_part = concat_binary[i+1]
        parent1_part = list(parent1_part)
        parent2_part = list(parent2_part)
        random_point = np.random.randint(0, 22)

        for j in range(random_point, 22):
            parent1_part[j], parent2_part[j] = parent2_part[j], parent1_part[j]

        parent1_part= ''. join(parent1_part)
        parent2_part= ''. join(parent2_part)
        result_list.append(parent1_part)
        result_list.append(parent2_part)
    return result_list
    
    

# which satisfies that future generation will be diversed by switching random individuals bit
def mutationOfBits(crossOver):
    value = crossOver[population-1]
    value = list(value)
    random_mutation = np.random.randint(0,22)

    if value[random_mutation]=='0':
        value[random_mutation]='1'
    else:
        value[random_mutation]='0'

    value = ''.join(str(v) for v in value)
    crossOver[population-1] = value

    return crossOver
    
# creation of next generation
def newPopTable(newlyCreatedPopulation, mutated_result, population):
    coordinates= []
    coordinates.append(newlyCreatedPopulation[population-1])
    coordinates.append(newlyCreatedPopulation[population-2])
    for i in range(2, population):
        # if newlyCreatedPopulation[population-1] not in coordinates:
        x = int(mutated_result[i][:11], 2)
        y = int(mutated_result[i][12:], 2)
        # -------------------------------------------------------
        if (x,y) in coordinates:
            while newlyCreatedPopulation[population - 3] in coordinates or population<2:
                population -= 1
            coordinates.append(newlyCreatedPopulation[population - 3])
            population -= 1
        else:
            if x <= groupImage.shape[0] and y <= groupImage.shape[1]:
                coordinates.append((x,y))
            else:    
                coordinates.append((np.random.randint(groupImage.shape[0]), np.random.randint(groupImage.shape[1])))
    # coordinates.append(newlyCreatedPopulation[population-1])
    return coordinates



# Calling functions
#####################################################################################
population = 100
fitnessThreshold = 0.85
currentGeneration = populationInitilization(groupColumns, groupRows, population)
fitnessVal = fitnessEvoluation(currentGeneration, groupImage, boothi)
selectedGeneration = selection(currentGeneration, fitnessVal[0])
testing = faceDetection(currentGeneration, fitnessVal, selectedGeneration)
currentGeneration2 = newGeneration(currentGeneration, selectedGeneration, fitnessVal[1])


count = 0
generations = [0]
save_max = []
save_max.append(selectedGeneration[population-1])
save_mean = []
save_mean.append(np.mean(selectedGeneration))
save_min = []
save_min.append(selectedGeneration[0])

# define No: of generation by changing the  range of for loop
for i in range(1, 500):
    if testing[1] == []:
        # print(testing[0][0][0])
        x,y = testing[0][0][0][0], testing[0][0][0][1]
        rect1 = cv2.rectangle(groupImage, (y, x), (y + boothiRows, x + boothiColumns), (255,0,0), 2)
        cv2.imshow('Region of Interest', rect1)
        cv2.waitKey(0)
        break
    else:
        generations.append(i)
        save_max.append(selectedGeneration[population-1])
        save_mean.append(np.mean(selectedGeneration))
        save_min.append(selectedGeneration[0])
        crossOver = crossOverOfBits(testing)
        mutation = mutationOfBits(crossOver)
        new_pop_table = newPopTable(testing, mutation, population)
        fitness_values = fitnessEvoluation(new_pop_table, groupImage, boothi)
        sorted_fitness = selection(new_pop_table, fitness_values[0])
        print(sorted_fitness[population-1])
        count=count+1
    testing = faceDetection(new_pop_table, fitness_values, sorted_fitness)
    if i == 499:
        print(testing[0])
        x,y = testing[0][0], testing[0][1]
        rect1 = cv2.rectangle(groupImage, (y, x), (y + boothiRows, x + boothiColumns), (255,255,0), 2)
        cv2.imshow('Region of Interest when worst case', rect1)
        cv2.waitKey(0)
        

    
    
print("Number of generations:", count)


# PLOT OF THE FITNESS VALUE
plt.plot(generations, save_max, label = 'max')
plt.plot(generations, save_mean, label = 'mean')
plt.plot(generations, save_min, label= 'min')
plt.title('Fitness against each generation')
plt.xlabel('No. of Generations')
plt.legend()
plt.show()

