import numpy as np
import matplotlib.pyplot as plt

k = 2
x = np.array([[0,-6],[4,4],[0,0],[-5,2]]).astype(float)
#x1 = np.random.randint(-10, 3, (100,2)).astype(float)
#x2 = np.random.randint(-5, 10, (100,2)).astype(float)
#x = np.concatenate((x1,x2),axis=0)
z = np.array([[0,-6],[-5,2]]).astype(float)

def l1_norm_element(x,z):
    return np.sum(x-z)

def compute_distance(x,z,distance_func=np.linalg.norm):
    if distance_func == 'l1':
        distance_func = np.sum
    else:
        distance_func = np.linalg.norm
    k = np.size(z,0)
    n = np.size(x,0)
    C = [[] for i in range(k)]
    cost = [[] for i in range(k)]
    distances = np.zeros(k)
    for i in range(n):
        for j in range(k):
            distances[j] = distance_func(abs(x[i]-z[j]))
        index = np.argmin(distances)
        C[index].append(i)
        cost[index].append(distances[index])
        total_cost = sum([item for sublist in cost for item in sublist])
    return cost, C, total_cost

def calculate_cost_matrix(x,distance_func=np.linalg.norm):
    if distance_func == 'l1':
        distance_func = np.sum
    else:
        distance_func = np.linalg.norm
    n = np.size(x,0)
    cost_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i,j] = distance_func(abs(x[i]-x[j]))
    return cost_matrix  

def update_z_medioid(C,x, distance_func):
    cost_matrix = calculate_cost_matrix(x,distance_func)
    z = np.zeros((len(C),np.size(x,1)))
    for j in range(len(C)):
        cost_vector = {}
        for i in C[j]:
            cost_vector[i] = 0
            for k in C[j]:
                cost_vector[i] += cost_matrix[i,k]
        min_cost = min(cost_vector, key=cost_vector.get)
        z[j] = x[min_cost]
    return z

def update_z_means(C,x):
    z = np.zeros((len(C),np.size(x,1)))
    for j in range(len(C)):
        x_sum = np.zeros(np.size(x,1))
        for i in range(len(C[j])):
            x_sum = np.add(x_sum,x[C[j][i]])
        z[j] = x_sum/len(C[j])
    return z

def run_algorithm(x,z, distance_func):
    prev_cost = 0
    delta_cost = 1
    while abs(delta_cost)>0.01:
        costs, C, total_cost = compute_distance(x,z, distance_func)
        z = update_z_means(C,x)
        #z = update_z_medioid(C,x, distance_func)
        print(total_cost)
        print(prev_cost)
        delta_cost = total_cost-prev_cost
        prev_cost = total_cost
        colours = {
            0: 'r',
            1: 'g',
            2: 'b',
            3: 'c',
        }
        for j in range(len(C)):
            for i in C[j]:
                plt.scatter(x[i,0], x[i,1], color = colours[j])
        #plt.scatter(x[:,0], x[:,1])
        plt.scatter(z[:,0], z[:,1], color='y')
        plt.show()
    
    print("zs are:\n",z)
    print("clusters are:\n", C)
    for j in range(len(C)):
            for i in C[j]:
                plt.scatter(x[i,0], x[i,1], color = colours[j])
        #plt.scatter(x[:,0], x[:,1])
    plt.scatter(z[:,0], z[:,1], color='y')
    plt.show()

distance_func = 'l2'
plt.scatter(x[:,0], x[:,1])
plt.scatter(z[:,0], z[:,1])
plt.show()
run_algorithm(x,z, distance_func)
# cost_matrix = calculate_cost_matrix(x)
# print(cost_matrix)
# costs, C, total_cost = compute_distance(x,z,'l1')
# z = update_z_medioid(C, x)
# print(z)
# plt.scatter(x[:,0], x[:,1])
# plt.scatter(z[:,0], z[:,1])
# plt.show()

# costs, C, total_cost = compute_distance(x,z,'l1')
# print(costs)
# print(C)
# print(len(C))
# print(total_cost)
# z = update_z_means(C,x)
# print(z)
# plt.scatter(x[:,0], x[:,1])
# plt.scatter(z[:,0], z[:,1])
# plt.show()