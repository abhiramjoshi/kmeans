import numpy as np
import scipy
import matplotlib.pyplot as plt
## k means algorithm

#generate zs

#partition vectors into groups
#update zs

#need group list
#define j_clust
#define k

#generate random coordinates

def init_z(x_vect, k):
    N = np.size(x_vect,0)
    z = np.zeros((k,2))
    choices = []
    for i in range(k):
        present = True
        while present == True:
            r = np.random.randint(0,N)
            if r not in choices:
                choices.append(r)
                present = False
        z[i] = x_vect[r]
    print(f'random numbers were: {choices}')
    return z

def group(x_vect,z):
    n = np.size(x_vect, 0)
    k = np.size(z,0)
    c = np.zeros((n))
    for xi in range(n):
        index = 0
        min_dist = np.linalg.norm(x_vect[xi]-z[0])
        for i in range(k):
            min_t = np.linalg.norm(x_vect[xi]-z[i])
            if min_t < min_dist:
                index = i
                min_dist = min_t
        c[xi] = index
    return c    

def group_g(c,k):
    G = [[] for i in range(k)]
    for i in range(np.size(c,0)):
        G[int(c[i])].append(i)
    return G

# def group_rep(G,x_vect):
#     n = int(np.size(x_vect,1))
#     k = len(G)
#     z = np.zeros(n,k)
def calc_j_clust(clusters,z):
    k = len(clusters)
    j = 0
    N_tot = 0
    for  i in range(k):
        clus = clusters[i][~np.all(clusters[i] == 0, axis=1)]
        #print(f'cluster is: {clus}')
        N = np.size(clus,0)
        #print(f' length of cluster is: {N}') 
        #print(f'cluster number is {i}')
        for j in range(N):
            sum_norm = np.linalg.norm(clus[j]-z[i])
        j = j + sum_norm
        N_tot = N_tot + N
    j = j/N
    return j

def group_clusters(c,x_vect,k):
    clusters = []
    for i in range(k):
        clusters.append(group_vectors(c,x_vect,i))
    return clusters

def group_vectors(c,x_vect, group):
    N = np.size(c,0)
    q = np.zeros((N,2))
    j = 0
    for i in range(N):
        if c[i] == group:
            q[j] = x_vect[i]
            j += 1
    
    return q

def mask(q):
    q = np.ma.masked_equal(q,0)
    return q

def create_rep(clusters):
    k = len(clusters)
    n = np.size(clusters[0],1)
    z = np.zeros((k,n))
    for i in range(k):
        z[i] = centroid(clusters[i])
    return z

def centroid(x):
    n = np.size(x,1)
    centr = np.zeros((1,n))
    x = x[~np.all(x == 0, axis=1)]
    x_sum = np.sum(x[:,0], axis = 0)
    y_sum = np.sum(x[:,1], axis = 0)
    centr[0,0] = x_sum/np.size(x,0)
    centr[0,1] = y_sum/np.size(x,0)
    return centr

def  remove_group(z,k):
    for i in range(k):
        if np.isnan(np.sum(z)):
            z[i] = [0,0]
    z1 = z[~np.all(z == 0, axis=1)]
    return z1         
def get_k(labels):
    l = []
    for i in labels:
        if i not in l:
            l.append(i)
    k = len(l)
    return k  

def k_means(feature_vector, labels):
    k = get_k(labels)
    j = 0
    z = init_z(feature_vector,k)
    conv_end = 0.00001
    conv = False

    while  conv == False: 
        # update G and c and clusters
        c = group(feature_vector,z)
        #G = group_g(c,k)
        clusters = group_clusters(c,feature_vector,k)
        j = calc_j_clust(clusters,z)
        #update z reps
        z2 = create_rep(clusters)
        z2 = remove_group(z2,k)
        j2 = calc_j_clust(clusters,z2)
        print(f'J_clust is {j}')
        for i in range(k):
            mask(clusters[i])
            plt.scatter(clusters[i][:,0],clusters[i][:,1])     
        plt.scatter(z[:,0],z[:,1], marker = '2', color = 'black', s = 200)
        plt.scatter(z2[:,0],z2[:,1], marker = '1', color = 'magenta', s = 200)
        plt.show()
        z = z2
        print(f'The clustering has improved by: {j-j2}')
        if abs(j-j2) < conv_end:
            conv = True

    print('complete')
    for i in range(k):
        mask(clusters[i])
        #plt.scatter(clusters[i][:,0],clusters[i][:,1], color = colours[i])
        plt.scatter(clusters[i][:,0],clusters[i][:,1])     
    plt.scatter(z[:,0],z[:,1], marker = '2', color = 'black', s = 200)
    #plt.scatter(z2[0],z2[1], marker = '1', color = 'black', s = 100)
    plt.scatter(z2[:,0],z2[:,1], marker = '1', color = 'magenta', s = 200)
    plt.show()

if __name__ == "__main__":
# create initial parameters
    k = 6
    N = np.random.randint(100,1000)
    x_vect = np.random.rand(N,2)
    j = 0
    z = init_z(x_vect,k)
    conv_end = 0.00001
    iteration = 0
    iteration_end = 10
    conv = False

    while  conv == False: 
        # update G and c and clusters
        c = group(x_vect,z)
        G = group_g(c,k)
        clusters = group_clusters(c,x_vect,k)
        j = calc_j_clust(clusters,z)
        #update z reps
        z2 = create_rep(clusters)
        z2 = remove_group(z2,k)
        j2 = calc_j_clust(clusters,z2)
        print(f'J_clust is {j}')
        iteration += 1
        for i in range(k):
            mask(clusters[i])
            #plt.scatter(clusters[i][:,0],clusters[i][:,1], color = colours[i])
            plt.scatter(clusters[i][:,0],clusters[i][:,1])     
        plt.scatter(z[:,0],z[:,1], marker = '2', color = 'black', s = 200)
        #plt.scatter(z2[0],z2[1], marker = '1', color = 'black', s = 100)
        plt.scatter(z2[:,0],z2[:,1], marker = '1', color = 'magenta', s = 200)
        plt.show()
        z = z2
        print(f'The clustering has improved by: {j-j2}')
        if abs(j-j2) < conv_end:
            conv = True

    print('complete')
    if np.size(x_vect,1) == 2:
        for i in range(k):
            mask(clusters[i])
            #plt.scatter(clusters[i][:,0],clusters[i][:,1], color = colours[i])
            plt.scatter(clusters[i][:,0],clusters[i][:,1])     
        plt.scatter(z[:,0],z[:,1], marker = '2', color = 'black', s = 200)
        #plt.scatter(z2[0],z2[1], marker = '1', color = 'black', s = 100)
        plt.scatter(z2[:,0],z2[:,1], marker = '1', color = 'magenta', s = 200)
        plt.show()


    #z2 = centroid(clusters[0])
    # c1 = group_vectors(c,x_vect,0)
    # c1 = np.ma.masked_equal(c1,0)
    # print(G)
    # print("--------")
    # print(c)
    # print("--------")
    # print(z)
    # print(np.size(z,0))
    # print("--------")
    #print(x_vect)
    #print("--------")
    #print(j,j2)
    #print(clusters)
    # print(x_vect[:,1])
    # print("--------")
    # print(clusters[0][:,1])
    colours = ['red','blue','green','yellow','magenta','cyan']


    #x2 = np.array([2,2,2])
    #x3 = np.sum(x_vect, axis = 0)

    #print(np.size(clusters[0][:,1],0))
    #c2 = clusters[0][clusters[0] != 0]
    #c2 = clusters[0][~np.all(clusters[0] == 0, axis=1)]
    # print("zeros removed -------")
    # print(c2)


    #print(z)
    #print('zeros removed--------------')
    #print(z2)
    #print('--------------')
    #print(z3)
    #print(np.size(z2,0))
    #plt.scatter(x_vect[:,0],x_vect[:,1])
    #plt.scatter(c1[:,0],c1[:,1], color = 'red')





