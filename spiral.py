import numpy as np
        
def spiral_mat_to_vect(A):
    v = []
    while(A.size != 0):
        #print(A)
        #print(A[0,:])
        v.append(A[0,:])
        A = A[1:,:].T[::-1]
    return np.concatenate(v)

def spiral_vect_to_mat(v):
    L = int(np.sqrt(v.size))
    l = L
    change_l = True
    A = np.zeros((L,L))
    i = 3
    x = 0
    y = 0
    #print("i={}, x={}, y={}".format(i,x,y))
    #print(A)
    A[x,y:l] = v[0:l]
    #print(A)
    #print("\n")
    A = A.T[::-1]

    v = v[l:len(v)]

    while(v.size != 0):
        i += 1
        if i % 2 == 0:
            l -= 1
        if (i + 1) % 4 == 0:
            x += 1
        if i % 4 == 0:
            y += 1
        #print("i={}, x={}, y={}".format(i,x,y))
        #print(A)
        A[x,y:y+l] = v[0:l]
        #print(A)
        #print("\n")
        A = A.T[::-1]
        v = v[l:len(v)]

        #print(A.T)
    #    A = A[1:,:].T[::-1]
    #return np.concatenate(out)
    for rotations in range(i % 4): 
        A = A.T[::-1]
    return A

L = 7 # matrice L x L

A = np.zeros((L,L))

num = 0  # La riempio coi numeri in fila
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        num += 1
        A[i,j] = num
v = spiral_mat_to_vect(A)
B = spiral_vect_to_mat(v)

print("Originale:")
print(A)
print("\n")

print("Vettore spirale:")
print(v)
print("\n")


print("Matrice ricostruita:")
print(B)
print("\n")
