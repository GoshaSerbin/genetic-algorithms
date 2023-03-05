import numpy as np
from time import time
import matplotlib.pyplot as plt

# РАНДОМНОЕ ЗАПОЛНЕНИЕ
W = 16 # грузоподъемность рюкзака
n = 60 # количество предметов
weights = np.random.rand(n) # веса предметов
values = np.random.rand(n) # ценности предметов

# ЗАПОЛНЕНИЕ ИЗ ФАЙЛА
weights = []
values = []
f = open('test2.txt', 'r')
n = int(f.readline())
for i in range(n):
    w, v = f.readline().split()
    weights.append(float(w))
    values.append(float(v))
W = float(f.readline())

# функция приспособленности
def fitness(x):
    if np.dot(weights, x) > W:
        return 0
    else:
        return np.dot(values, x)

# кроссинговер
def cross(x1, x2):
    p = np.random.randint(1, n-1) # точка скрещивания
    return [x1[:p]+x2[p:], x2[:p]+x1[p:]]

#однородный кроссинговер
def homo_cross(x1, x2):
    mask = np.random.choice(n, size = n//2, replace = False) 
    y1 = [x1[i] if i in mask else x2[i] for i in range(n)]
    y2 = [x2[i] if i in mask else x1[i] for i in range(n)]
    return [y1, y2]

def mutation(x):
    indices = np.random.choice(n, size = 2, replace = False)  
    return [(1 - x[i]) if i in indices else x[i] for i in range(n)]

def inversion(x):
    return x[::-1]

pop_size = 20 # численность популяции
pop = np.random.randint(2, size=(pop_size, n)).tolist()

N = 100 # число поколений
pc, pm, pi = 0.9, 0.1, 0.01 # вероятности кроссовера, мутации, инверсии

results_max = []
results_avg = []
t = -time()
for generation in range(N):
    fits = [fitness(pop[i]) for i in range(pop_size)]
    avg_fit = sum(fits) / pop_size
    results_avg.append(avg_fit)
    results_max.append(max(fits))
    print('Средняя приспособленность поколения', generation,' равна', avg_fit)
    new_pop = []
    pop.sort(key=fitness)
    pop = pop[-pop_size//2:] # отбор на основе усечения
    while len(new_pop) < pop_size:
        choice = np.random.choice(len(pop), 2, replace = False)
        parents = [pop[i] for i in choice]    
        new_pop.extend(parents)
        if np.random.rand() < pc:
            children = homo_cross(parents[0], parents[1])
            for i in range(2):            
                if np.random.rand() < pm:
                    children[i] = mutation(children[i])
                if np.random.rand() < pi:
                    children[i] = inversion(children[i]) 
                if fitness(children[i]) > avg_fit: # элитарная схема
                    new_pop.append(children[i])     
    pop = new_pop.copy()
t += time()

fits = [fitness(pop[i]) for i in range(pop_size)]
avg_fit = sum(fits) / pop_size
results_avg.append(avg_fit)
results_max.append(max(fits))

print('Время:', t)
ans = pop[fits.index(results_max[-1])]
print('Ответ:', *ans)
print('Общая ценность взятых предметов:', results_max[-1])
print('Общий вес взятых предметов:', np.dot(weights, ans))
fig = plt.figure(figsize = (8,5))
ax = fig.add_subplot()
ax.set_title('Задача о рюкзаке', fontsize = 16, fontname="Times New Roman")
ax.set_xlabel('Поколение', fontsize = 16, fontname="Times New Roman")
ax.set_ylabel('Ценность', fontsize = 16, fontname="Times New Roman")
ax.plot(results_max,'-b', label = 'лучшая по популяции') 
ax.plot(results_avg,'--y',label = 'средняя по популяции') 
ax.plot([0, N],[89, 89],'--', color = (0,0,0,0.5),label = 'лучшее решение')
ax.legend(loc = 'right')
plt.grid()
plt.show()