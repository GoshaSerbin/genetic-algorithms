import numpy as np
from time import time
import matplotlib.pyplot as plt

# РАНДОМНОЕ ЗАПОЛНЕНИЕ
n = 3 # количество неизвестных
# max_number = 1000
# coeffs = np.random.randint(1, max_number, size = n) # коэффициенты уравнения
# d = np.random.randint(1, 100*max_number) # правая часть

coeffs = [3, 2, 1]
d = 25
max_number = 20

# функция приспособленности
def fitness(x):
    return abs(d - np.dot(coeffs, x))

# кроссинговер
def cross(x1, x2):
    p = np.random.randint(1, n-1) # точка скрещивания
    return [x1[:p]+x2[p:], x2[:p]+x1[p:]]

# однородный кроссинговер
def homo_cross(x1, x2):
    mask = np.random.choice(n, size = n//2, replace = False) 
    y1 = [x1[i] if i in mask else x2[i] for i in range(n)]
    y2 = [x2[i] if i in mask else x1[i] for i in range(n)]
    return [y1, y2]

def swap_mutation(x):
    indices = np.random.choice(n, size = 2, replace = False)  
    new_x = x.copy()
    new_x[indices[0]] = x[indices[1]]
    new_x[indices[1]] = x[indices[0]]
    return new_x

# случайный сброс
def reset_mutation(x):
    indices = np.random.choice(n, size = 2, replace = False)  
    return [np.random.randint(1, max_number) if i in indices else x[i] for i in range(n)]

def inversion(x):
    return x[::-1]

pop_size = 5 # численность популяции
pop = np.random.randint(1, max_number, size=(pop_size, n)).tolist()

N = 100 # число поколений
pc, pm, pi = 0.9, 0.1, 0.01 # вероятности кроссовера, мутации, инверсии

results_max = []
results_avg = []
t = -time()
for generation in range(N):
    fits = [fitness(pop[i]) for i in range(pop_size)]
    if min(fits) == 0:
        break
    s = sum([1/fits[i] for i in range(pop_size)])
    probabilities = [1 / (fits[i] * s) for i in range(pop_size)]
    avg_fit = sum(fits) / pop_size
    results_avg.append(avg_fit)
    results_max.append(min(fits))
    print('Средняя приспособленность поколения', generation,' равна', avg_fit)
    new_pop = []
    while len(new_pop) < pop_size:
        choice = np.random.choice(pop_size, 2, replace = False, p = probabilities) # метод пропорционального отбора
        parents = [pop[i] for i in choice]    
        new_pop.extend(parents)
        if np.random.rand() < pc:
            children = homo_cross(parents[0], parents[1])
            child_fit = sum([fitness(children[i]) for i in range(2)])/2
            parent_fit = sum([fitness(parents[i]) for i in range(2)])/2
            if child_fit >= parent_fit:
                for i in range(2):            
                    if np.random.rand() < pm:
                        children[i] = reset_mutation(children[i])
                    if np.random.rand() < pi:
                        children[i] = inversion(children[i])          
            new_pop.extend(children)
    pop = new_pop.copy()
t += time()

fits = [fitness(pop[i]) for i in range(pop_size)]
avg_fit = sum(fits) / pop_size
results_avg.append(avg_fit)
results_max.append(min(fits))

print('Время:', t)
print('Коэффициенты уравнения:', *coeffs,'=', d)
ans = pop[fits.index(results_max[-1])]
print('Ответ:', *ans)
print('Невязка:', results_max[-1])
fig = plt.figure(figsize = (8,5))
ax = fig.add_subplot()
ax.set_title('Решение уравнения', fontsize = 16, fontname="Times New Roman")
ax.set_xlabel('Поколение', fontsize = 16, fontname="Times New Roman")
ax.set_ylabel('Невязка', fontsize = 16, fontname="Times New Roman")
ax.plot(results_max,'-b', label = 'лучшая по популяции') 
ax.plot(results_avg,'--y',label = 'средняя по популяции') 
ax.legend(loc = 'right')
plt.grid()
plt.show()