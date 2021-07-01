# -*- coding: utf-8 -*-

import numpy as np
import math
import scipy.linalg
import time
import matplotlib
import matplotlib.pyplot as plt

def det2x2(m):                                                                  # Check that the matrix is 2x2.
    if m.shape[0] != 2 or m.shape[1] != 2:
        raise Exception("Non2x2Matrix")
     
    return (m[0][0]*m[1][1] ) - ( m[0][1]*m[1][0])                              # Calculate determinant of 2x2 matrix.

def inverse2x2(m):                                                              # Check that matrix is 2x2.
    if (m.shape[0] != 2 or m.shape[1] != 2):
        raise Exception("Non2x2Matrix")

    det = det2x2(m)                                                             # Calculate inverse of 2x2.
    tmp = np.empty([2,2])
    tmp[0][0] = m[1][1] / det
    tmp[0][1] = -1 * m[0][1] / det
    tmp[1][0] = -1 * m[1][0] / det
    tmp[1][1] = m[0][0] / det
    return tmp

def submatrix(m,i,j):                                                           # Check that matrix is square.
    if not (m.shape[0] == m.shape[1]):
        raise Exception("NonSquareMatrix")
    
    n = m.shape[0]-1                                                            # Create new matrix of order n-1.
    tmp = np.empty([n,n])
                                                                               
    tmpi = list(range(m.shape[0]))                                              # Delete rows and columns to create cofactors.
    del tmpi[i]
    tmpj = list(range(m.shape[1]))
    del tmpj[j]
    
    for ii in range(n):
        for jj in range(n):
            tmp[ii][jj] = m[tmpi[ii]][tmpj[jj]]

    return tmp

def det(m):
    if not (m.shape[0] == m.shape[1]):
        raise Exception("NonSquareMatrix")
    
    if (m.shape[0]==2):                                                         # Determinant for 2x2 matrix.
        return det2x2(m)
        
    else:
        tmp = 0
        
        for j in range(m.shape[0]):                                             # Use 0-th row to calculate determinant.
            tmp = tmp + (math.pow(-1,j) * m[0][j] * det(submatrix(m,0,j)))
    
    return tmp

def adjugate(m):
    if not (m.shape[0] == m.shape[1]):
        raise Exception("NonSquareMatrix")
    
    tmp = np.empty(m.shape)                                                     # Create empty matrix of order n.
    
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            tmp[j][i] = (math.pow(-1,i+j) * det(submatrix(m,i,j)))              # Note order of indices.
    return tmp

def inverse(m):
    if not (m.shape[0] == m.shape[1]):
        raise Exception("NonSquareMatrix")
    
    if m.shape[0]==2:
        return inverse2x2(m)
    
    d = det(m)        
    c = adjugate(m)
    return np.true_divide(c,d)                                                  # Calculate inverse of non-2x2.

###############################################################################
"""
def mround(matrix, dp):                                                         # Rounding the values in the matrix to account for error due to limited floating point precision.
    for i in range(matrix.shape[0]):                                           
        for j in range(matrix.shape[1]):
            matrix[i][j] = round(matrix[i][j], dp)
    return matrix

a = np.random.rand(4, 4)                                                        # 4x4 matrix of random numbers between 0 and 1.
y = np.random.rand(4, 1)
     
print(a)

b = np.dot(inverse(a), y)
c = scipy.linalg.inv(a)                                                         # Scipy linalg matrix inversion.

lu, piv = scipy.linalg.lu_factor(a)
lu_solution = scipy.linalg.lu_solve((lu, piv), y)

print(mround(b, 2))
print(lu_solution)
"""
###############################################################################
"""
timeC = []                                                                      # Defining the time arrays.
timeL = []
timeS = []
NC = []                                                                         # Defining the N arrays (one for Cramer and one for LU and SVD).
N = []

def SingCheck(m):                                                               # Defining a function that checks if the randomly generated matrix is singular.
    if scipy.linalg.det(m) == 0:
        raise Exception("Singular matrix!")

for i in range(2, 600):                                                         # Filling the N arrays with values.
    N.append(i)
for i in range(2, 8):
    NC.append(i)
    
for i in NC:
    m = np.random.rand(i, i)                                                    # Generating the random matrices and solutions.
    y = np.random.rand(i, 1)
    SingCheck(m)
    
    startC = time.process_time()
    solC = np.dot(inverse(m), y)                                                # Calulating the solutions for Cramer.
    timeC.append(time.process_time() - startC)

for i in N:
    m = np.random.rand(i, i)
    y = np.random.rand(i, 1)
    SingCheck(m)
        
    startL = time.process_time()
    lu, piv = scipy.linalg.lu_factor(m)                                         # Calculating the solutions for LU.
    solL = scipy.linalg.lu_solve((lu, piv), y)                               
    timeL.append(time.process_time() - startL)
    
    startS = time.process_time()
    U,s,Vh = scipy.linalg.svd(m)                                                # Calculating the solutions for SVD.
    first = np.dot(U.T, y)
    second = np.dot(np.diag(1/s), first)
    solS =  np.dot(Vh.conj().T, second)
    timeS.append(time.process_time() - startS)
  
#plt.plot(NC, timeC, label = str("Cramer's rule"))                               # Plotting Cramer's rule versus time.
plt.plot(N, timeL, label = str("LU decomposition"))                             # Plotting LU and SVD versus time.
plt.plot(N, timeS, label = str("SVD decomposition"))
plt.xlabel("N")
plt.ylabel("Time (s)")
plt.legend()
plt.show() 
"""
###############################################################################

RHS = np.array([5,10,15])                                                       # Defining the right hand side matrix.
A = []                                                                          # Defining the left hand side matrix.  
k = []                                                                          # Creating a list that will store the values of k.
Sol_C = []                                                                      # Creating lists that will store the x, y, and z solutions for the three methods.
Sol_L = []
Sol_S = []
decomposition = []                                                              # Creating lists that are needed to calculate the solutions for each method.
U = []
s = []
Vh = []
first = []
second = []
p_C = []                                                                        # Creating lists that will store the precisions for each method.
p_L = []
p_S = []

for i in range(1000):                                                           # 1000 data points.
    
    k.append(10**(-i/100))                                                      # k varies between 0 and 10.
    A.append(np.array([[1,1,1],[1,2,-1],[2,3,k[i]]]))                           # Creating A matrices for each k value.
    Sol_C.append(np.dot(inverse(A[i]), RHS))                                    # Calculating the solutions for each method.
    
    decomposition.append(scipy.linalg.lu_factor(A[i]))                                  
    Sol_L.append(scipy.linalg.lu_solve(decomposition[i], RHS))
    
    U1,s1,Vh1 = scipy.linalg.svd(A[i])
    U.append(U1)
    s.append(s1)
    Vh.append(Vh1)                                               
    first.append(np.dot(U[i].T, RHS))
    second.append(np.dot(np.diag(1/s[i]), first[i]))
    Sol_S.append(np.dot(Vh[i].conj().T, second[i]))
    
    p_C.append(abs((Sol_C[i][0] + Sol_C[i][1] + Sol_C[i][2]) - 5))              # Calculating the precisions for each method.
    p_L.append(abs((Sol_L[i][0] + Sol_L[i][1] + Sol_L[i][2]) - 5))
    p_S.append(abs((Sol_S[i][0] + Sol_S[i][1] + Sol_S[i][2]) - 5))
    
plt.scatter(k,p_C, label = str("Cramer"), s = 1)                                # Plotting the precision of each method versus k.
plt.loglog(k,p_L, label = str("LU"))
plt.loglog(k,p_S, label = str("SVD"))
plt.xlabel("log(k)")
plt.ylabel("log(Error)")
plt.legend()
plt.show()

###############################################################################
"""
def eigen(A, mu0, b0, precision):                                               # Taking the matrix, initial eigenvalue and vector, and precision as arguments.
    mu1 = 0                                                                     
    I = np.identity(A.shape[0])                                                 # Setting the identity matrix to be equal in dimension to A.
    difference = abs(mu1 - mu0)                                                 # Defining the difference between the eigenvalues to break out of the loop.
    
    while difference > precision:
        mu1 = mu0
        AI = A - mu1 * I
        InvAI = scipy.linalg.inv(AI)
        b0 = np.dot(InvAI, b0)/scipy.linalg.norm(np.dot(InvAI, b0))             # Calculating the new eigenvector.
        mu0 = np.dot(np.dot(np.conj(b0), A), b0)/np.dot(np.conj(b0), b0)        # Calculating the new eigenvalue.
        difference = abs(mu1 - mu0)                                             # Resetting the difference.
    
    return mu0, b0                                                              # Returning the final eigenvalue and vector.
                                                       
A = np.array([[1,2,3],[1,2,1],[3,2,1]])                                         # Defining A - can be any size as long as it's square.

if not (A.shape[0] == A.shape[1]):                                              # Checks that A is square.
        raise Exception("Non Square Matrix")
if scipy.linalg.det(A) == 0:                                                    # Checks that A is not singular.
    raise Exception("Singular matrix!")

mu0 = 200                                                                       # Setting the initial estimate of the eigenvalue.
b0 = np.array([1,1,1])                                                          # Setting the inital estimate of the eigenvector.
if not (b0.shape[0] == A.shape[0]):                                             # Checks that the initial eigenvector estimate is of the right dimension.
    raise Exception("Eigenvalue must be the same length as dimension of A")

precision = 1e-20                                                               
        
print(eigen(A, mu0, b0, precision))
"""
###############################################################################   
"""
def normode(wall0, k, m, precision):                                            # Taking the system (walls or no walls), k, m, and the precision as arguments.
    
    wall1 = k/m * wall0
    
    e_vals, e_vecs = scipy.linalg.eig(wall1)                                    # Using scipy to find the eigenvalues and eigenvectors.
    
    for i in range(2):
        for j in range(2):
            if e_vecs[i][j] < precision and e_vecs[i][j] > -precision:          # Setting the eigenvector values to 0 if they are very close to 0 due to rounding error.
                e_vecs[i][j] = 0
             
    e_vecs_split = np.split(e_vecs, 3)                                          # Splitting the eigenvector array into three seperate arrays.
                
    print("Frequencies: ",np.sqrt(abs(float(e_vals[0]))),", " ,np.sqrt(abs(float(e_vals[1]))),", " ,np.sqrt(abs(float(e_vals[2]))))
    print("Normal modes: ",e_vecs_split)                                        # Taking the square root of each eigenvector, since this gives the frequency.
    
wallYes = np.array([[-2,1,0],[1,-2,1],[0,1,-2]])                                # Matrix for the case with walls.
wallNo = np.array([[-1,1,0],[1,-2,1],[0,1,-1]])                                 # Matrix for the case without walls.

normode(wallYes, 1, 1, 10**(-10))                                               # Setting k and m to 1 for simplicity.
normode(wallNo, 1, 1, 10**(-10))    
"""    
    
    



































                                         

