import numpy as np

import joblib #Used only when save_to_disk = True
import os  #Used only when save_to_disk = True



def rbf_matrix_fit(x, gamma):
    #This will split the matrix (of shape (x,x)) on 9 blocks and calculate the rbf kernel for the 6 lower half blocks with iterations.
    # On each iteration it will copy the transposed block on the mirrored one using the fact that the matrix is symmetric
    N = x.shape[0]
    if N <= 2000:
        block_size = max(1, N // 3) #ensure that is one when x<3. The 9 blocks i found ran faster on the iris dataset, so i picked it
    else:
        block_size = max(1, N // 6) #If N is large use smaller block size to reduce peak memory consumption
    result = np.zeros((N, N), dtype=np.float32)
    x = x.astype(np.float32)  # Convert inputs to float32
    gamma = np.float32(gamma)
    for i in range(0, N, block_size):
        for j in range(i, N, block_size):
            # Compute block indices
            x_block_i = x[i:i+block_size]
            x_block_j = x[j:j+block_size]
            # Compute pairwise differences for the current block
            diff = x_block_i[:, np.newaxis, :] - x_block_j[np.newaxis, :, :] #Vectorized calculations of differences on this block
            squared_diff = np.sum(diff**2, axis=2, dtype=np.float32)
            exp_block = np.exp(-gamma * squared_diff, dtype=np.float32)
            # Assign to the result matrix
            result[i:i+block_size, j:j+block_size] = exp_block
            if i != j:  # Mirror for symmetric positions
                result[j:j+block_size, i:i+block_size] = exp_block.T
    return result

def kernelized_dot_product_fit(x1, x2, kernel, d, gamma):
    if kernel == "Linear":
        dot_matrix = np.dot(x1, x2.T)
        return dot_matrix
    elif kernel == "Polynomial":
        dot_matrix = (np.dot(x1, x2.T) + 1)**d #Polynomial kernel of power d
        return dot_matrix
    elif kernel == "RBF":
        dot_matrix = rbf_matrix_fit(gamma=gamma, x=x1)
        return dot_matrix.astype(np.float64) #Convert to float64 again to avoid mixed precision calculations
    
def rbf_pred_func(x_train, x_test, gamma):
    M, N = x_train.shape
    m = x_test.shape[0]

    x_train = x_train.astype(np.float32)  # Convert inputs to float32
    x_test = x_test.astype(np.float32)
    gamma = np.float32(gamma)

    final_result = np.zeros((M, m), dtype=np.float32) 

    chunk_size = max(1, m // 3)
    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        x_test_chunk = x_test[start:end, :].astype(np.float32) 

        
        diff = x_train[:, np.newaxis, :] - x_test_chunk[np.newaxis, :, :]  # Shape (M, chunksize, N)
        squared_diff = np.sum(diff**2, axis=2, dtype=np.float32)  # Sum along the N dimension to get (M, chunksize)
        final_result[:, start:end] = np.exp(-gamma*squared_diff, dtype=np.float32)

    return final_result.astype(np.float64) #Convert back to float64 to avoid mixed presicion calculations

def symmetrize_matrix(matrix, n):
    #Calculates 0.5 * (matrix + matrix.T) incrementally to avoid the need of a 2XnXn memory allocation.
    for i in range(n):
        for j in range(i, n):  # Iterate over the upper triangle, including the diagonal
            avg_value = 0.5 * (matrix[i, j] + matrix[j, i]) 
            matrix[i, j] = avg_value
            matrix[j, i] = avg_value   #Make use of the symmetry
    
    return matrix

def compute_centering_in_chunks(kernel_matrix, n_samples, left, middle=False):
    #Calculates the three components needed for kernel centering  Im @ kernel_matrix, kernel_matrix @ Im,  Im @ kernel_matrix @ Im  without the need to  have Im in memory.
    chunk_size = max(1, n_samples // 6)
    final_matrix = np.zeros((n_samples, n_samples))
    if left==True and middle==False:
        for i in range(0, n_samples, chunk_size):
        
            chunk_end = min(i + chunk_size, n_samples)
            Im_chunk = np.full((chunk_end-i, n_samples), 1/n_samples)

            final_matrix[i:chunk_end, :] = Im_chunk @ kernel_matrix



        return final_matrix  #Im @ kernel_matrix
    elif left==False and middle==False:
        
        for i in range(0, n_samples, chunk_size):
        
            chunk_end = min(i + chunk_size, n_samples)

            Im_chunk = np.full((n_samples, chunk_end-i), 1/n_samples)

            final_matrix[:, i:chunk_end] = kernel_matrix @ Im_chunk

        return final_matrix  #kernel_matrix @ Im
            

        
    elif left==None and middle==True:
        for i in range(0, n_samples, chunk_size):
        
            chunk_end = min(i + chunk_size, n_samples)

            Im_chunk = np.full((n_samples, chunk_end-i), 1/n_samples)

            final_matrix[:, i:chunk_end] = compute_centering_in_chunks(kernel_matrix, n_samples, left=False, middle=False) @ Im_chunk
        return final_matrix  #Im @ kernel_matrix @ Im 

def center_kernel_fit(kernel_matrix, MemoryProblem):
    #Centers the kernel matrix. If MemoryProblem is True it does this without storing Im and ensures symmetry incrementally. If not it calculates Im and does the calculations.
    n_samples = kernel_matrix.shape[0]
    if MemoryProblem:
        kernel_matrix -= compute_centering_in_chunks(kernel_matrix, n_samples, left=True)
        kernel_matrix -= compute_centering_in_chunks(kernel_matrix, n_samples, left=False)
        kernel_matrix += compute_centering_in_chunks(kernel_matrix, n_samples, left=None, middle=True)  
        return symmetrize_matrix(matrix=kernel_matrix, n=n_samples)
    else:  
        Im = np.full((n_samples, n_samples), 1/n_samples)   # Matrix with dimensions n_samples X n_samples, with every value = 1/n_sample 
        kernel_matrix -= Im @ kernel_matrix
        kernel_matrix -= kernel_matrix @ Im
        kernel_matrix += Im @ kernel_matrix @ Im  #Kernel matrix centered in kernel feature space
        return 0.5 * (kernel_matrix + kernel_matrix.T)  #Ensure that is symmetric
    
def compute_centering_in_chunks_transform(kernel_matrix, n_samples_train, n_samples_test, k_test, first, second, third):
    #Calculates the three components needed for TEST kernel centering   Im_test @ kernel_matrix, k_test @ Im_train,  Im_test @ kernel_matrix @ Im_train  without the need to have Im_train and Im_test in memory.
    #All returns are matrices of shape (n_test X n_train)
    chunk_size = max(1, n_samples_test // 6)
    if first==True and second==False and third==False:
        Im_test_kernel_matrix = np.zeros((n_samples_test, n_samples_train))
        for i in range(0, n_samples_test, chunk_size):
        
            chunk_end = min(i + chunk_size, n_samples_test)
            Im_chunk = np.full((chunk_end-i, n_samples_train), 1/n_samples_train)
            Im_test_kernel_matrix[i:chunk_end, :] = Im_chunk @ kernel_matrix

        return Im_test_kernel_matrix  #Im_test @ kernel_matrix
    elif first==False and second==True and third==False:
        k_test_Im_train = np.zeros((n_samples_test, n_samples_train))
        for i in range(0, n_samples_train, chunk_size):
        
            chunk_end = min(i + chunk_size, n_samples_train)

            Im_chunk = np.full((n_samples_train, chunk_end-i), 1/n_samples_train)

            k_test_Im_train[:, i:chunk_end] = k_test @ Im_chunk

        return k_test_Im_train   #k_test @ Im_train
    elif first==False and second==False and third==True:
        Im_test_kernel_matrix_Im_train = np.zeros((n_samples_test, n_samples_train))
        for i in range(0, n_samples_train, chunk_size):
        
            chunk_end = min(i + chunk_size, n_samples_train)

            Im_chunk = np.full((n_samples_train, chunk_end-i), 1/n_samples_train)

            Im_test_kernel_matrix_Im_train[:, i:chunk_end] = compute_centering_in_chunks_transform(kernel_matrix, n_samples_train, n_samples_test, k_test, first=True, second=False, third=False) @ Im_chunk

        return Im_test_kernel_matrix_Im_train  #Im_test @ kernel_matrix @ Im_train

        

def center_kernel_transform(kernel_matrix, train_samples, test_samples, k_test, MemoryProblem):
    #Centers the TEST kernel matrix. If MemoryProblem is True it does this without storing Im_train and Im_test and ensures symmetry incrementally. If not it calculates Im_train and Im_test and does the calculations.
    if MemoryProblem:
        k_test -= compute_centering_in_chunks_transform(kernel_matrix=kernel_matrix, n_samples_train=train_samples, n_samples_test=test_samples, k_test=k_test, first=True, second=False, third=False)
        k_test -= compute_centering_in_chunks_transform(kernel_matrix=kernel_matrix, n_samples_train=train_samples, n_samples_test=test_samples, k_test=k_test, first=False, second=True, third=False)
        k_test += compute_centering_in_chunks_transform(kernel_matrix=kernel_matrix, n_samples_train=train_samples, n_samples_test=test_samples, k_test=k_test, first=False, second=False, third=True)
    else:
        Im_train = np.full((train_samples, train_samples), 1/train_samples)
        Im_test = np.full((test_samples, train_samples), 1/train_samples)
        Im_test = Im_test @ kernel_matrix  #Avoid recalculating repeating multiplications.
        k_test -= Im_test
        k_test -= k_test @ Im_train
        k_test += Im_test @ Im_train # Shape (number of test samples X number of train samples). Centered in kernel feature space. 
    return k_test #Centered test kernel matrix in feature space. Shape (n_test X n_train)

class KPCA:   
    #Performs kernel PCA. 
    #MemoryProblem  affects the way the kernel matrixes (train and test) centering is done by the functions center_kernel_fit, center_kernel_transform
    #If save_to_disk is true it saves the centered kernel matrix from training data (in fit and fit transform) on disk and then loads it during transform when needed. It helps control memory consumption.
    def __init__(self, kernel, d, gamma, n_components, MemoryProblem=False, save_to_disk=False):
        if gamma is not None and gamma<=0:
            raise ValueError("gamma must be positive.")
        if (kernel is not None) and (kernel not in ["Linear", "Polynomial", "RBF"]):
            raise ValueError("kernel must be a str with values ('Linear', 'Polynomial' or 'RBF')")
        if d is not None and d<2:
            raise ValueError("d must be d>=2.")
        
        if (kernel == "RBF") and (gamma==None):
            raise ValueError("Set gamma")
        if (kernel == "Polynomial") and (d is None):
            raise ValueError("Set d")
        
        self.n_components = n_components
        self.kernel = kernel
        self.d = d
        self.gamma = gamma
        self.MemoryProblem = MemoryProblem  
        self.save_to_disk = save_to_disk
        
        #Learned after fit
        self.needed_eigenvectors = None   
        self.kernel_matrix_fit = None
        self.n_train = None

    def fit(self, x):
        self.n_train = x.shape[0] 
        if self.save_to_disk:
            kernel_matrix = center_kernel_fit(kernelized_dot_product_fit(x1=x, x2=x, kernel=self.kernel, d=self.d, gamma=self.gamma), MemoryProblem=self.MemoryProblem)
            print(f"Kernel centered")
            self.kernel_matrix_fit = f"kernel_matrix_{self.kernel}.joblib" #Use self.kernel_matrix_fit to store the file name of the saved kernel matrix in disk instead of the kernel_matrix itself
            joblib.dump(kernel_matrix, self.kernel_matrix_fit) #Save it in disk        
            eigenvalues, eigenvectors = np.linalg.eigh(kernel_matrix)
            sorted_indeces = np.argsort(-eigenvalues)   

            eigenvectors = eigenvectors[:, sorted_indeces[0:self.n_components]]
            eigenvalues = eigenvalues[sorted_indeces[0:self.n_components]]   

            self.needed_eigenvectors = eigenvectors * (1/np.sqrt(eigenvalues))
        
        else:    
            self.kernel_matrix_fit = center_kernel_fit(kernelized_dot_product_fit(x1=x, x2=x, kernel=self.kernel, d=self.d, gamma=self.gamma), MemoryProblem=self.MemoryProblem) # Centered kernel matrix in feature space. kernelized_dot_product_fit calculates the kernel_matrix
            print(f"Kernel centered")
            eigenvalues, eigenvectors = np.linalg.eigh(self.kernel_matrix_fit)
            sorted_indeces = np.argsort(-eigenvalues)   #Sort indeces in a descenting way of eigenvalues

            eigenvectors = eigenvectors[:, sorted_indeces[0:self.n_components]] #Matrix of shape (n_train X n_components). The eigenvectors are in the columns.
            eigenvalues = eigenvalues[sorted_indeces[0:self.n_components]]   # Keep only the higher self.n_components eigenvalues and its corresponding eigenvectors

            self.needed_eigenvectors = eigenvectors * (1/np.sqrt(eigenvalues))  #Scale them so the transformation uses orthonormal eigenvectors.
        return self

    def transform(self, x_fit, x_test):
        n_test = x_test.shape[0]
        if self.kernel=='RBF':
            k_test = rbf_pred_func(x_train=x_fit, x_test=x_test, gamma=self.gamma).reshape(n_test,self.n_train)   # Shape (number of test samples X number of train samples). Test kernel matrix.
        else:
            k_test = kernelized_dot_product_fit(x1=x_fit, x2=x_test, gamma=self.gamma, d=self.d, kernel=self.kernel).reshape(n_test,self.n_train)
        print(f"Test kernel calculated")
        if self.save_to_disk:
            self.kernel_matrix_fit = joblib.load(self.kernel_matrix_fit)   #If save_to_disk is True load the kernel_matrix calculated in fit to use in the centering of the k_test.
        return center_kernel_transform(kernel_matrix=self.kernel_matrix_fit, train_samples=self.n_train, test_samples=n_test, k_test=k_test, MemoryProblem=self.MemoryProblem) @ self.needed_eigenvectors # Shape (number of test samples X n_components). Transformed x_test

    def fit_transform(self, x):
        self.n_train = x.shape[0] 
        if self.save_to_disk:
            kernel_matrix = center_kernel_fit(kernelized_dot_product_fit(x1=x, x2=x, kernel=self.kernel, d=self.d, gamma=self.gamma), MemoryProblem=self.MemoryProblem)
            print(f"Kernel centered")
            self.kernel_matrix_fit = f"kernel_matrix_{self.kernel}.joblib"
            joblib.dump(kernel_matrix, self.kernel_matrix_fit)
            eigenvalues, eigenvectors = np.linalg.eigh(kernel_matrix)
            sorted_indeces = np.argsort(-eigenvalues)   #Sort indeces in a descenting way 
            eigenvectors = eigenvectors[:, sorted_indeces[0:self.n_components]]
            eigenvalues = eigenvalues[sorted_indeces[0:self.n_components]]   # Keep only the higher self.n_components eigenvalues and its corresponding eigenvectors
            self.needed_eigenvectors = eigenvectors * (1/np.sqrt(eigenvalues))
            return kernel_matrix @ self.needed_eigenvectors
        
        else:    
            self.kernel_matrix_fit = center_kernel_fit(kernelized_dot_product_fit(x1=x, x2=x, kernel=self.kernel, d=self.d, gamma=self.gamma), MemoryProblem=self.MemoryProblem) # Centered kernel matrix in feature space
            print(f"Kernel centered")
            eigenvalues, eigenvectors = np.linalg.eigh(self.kernel_matrix_fit)
            sorted_indeces = np.argsort(-eigenvalues)   #Sort indeces in a descenting way 

            eigenvectors = eigenvectors[:, sorted_indeces[0:self.n_components]]
            eigenvalues = eigenvalues[sorted_indeces[0:self.n_components]]   # Keep only the higher self.n_components eigenvalues and its corresponding eigenvectors

            self.needed_eigenvectors = eigenvectors * (1/np.sqrt(eigenvalues))  #Scale them so the transformation uses orthonormal eigenvectors.
            return self.kernel_matrix_fit @ self.needed_eigenvectors #Return the transformation instead of self
        
    def cleanup_kernel_matrix_disk(self):
        #Deletes the kernel_matrix_fit created during training when save_to_disk==True.    
        try:
            filename = f"kernel_matrix_{self.kernel}.joblib"
            os.remove(filename)
            print(f"Deleted file: {filename}")
        except FileNotFoundError:
            print(f"File not found: {filename}")
        except Exception as e:
            print(f"Error deleting file {filename}: {e}") 
        print("Cleanup completed.")
        


class LDA:   
    def __init__(self, number_of_lda_components=None):
        if (number_of_lda_components != None) and (number_of_lda_components<1):
            raise ValueError("number_of_lda_components should be an int between 1-(Number of classes)-1")
        self.number_of_lda_components = number_of_lda_components

        #Learned after fit
        self.needed_eigenvectors = None
          

    def fit(self, x, y):
        uniques = np.unique(y)
        n_features = x.shape[1]
        max_rank = min(len(uniques)-1, n_features)  #Max rank of SW-1*SB matrix (see comments bellow for why)
        if (self.number_of_lda_components != None) and (self.number_of_lda_components>max_rank):
            raise ValueError("n_components should be an int between 1 and (Number of classes)-1")  #Raise error if number_of_lda_components>max_rank because there will be not so many (non zero) eigenvalues.
        overall_mean = np.mean(x, axis=0)  #Mean vector of all training data
        SW = np.zeros((n_features, n_features), dtype=float)
        SB = np.zeros((n_features, n_features), dtype=float) #Both matrixes have shape (Number of Dimensions X Number of Dimensions)
        for class_label in uniques:  
            #For every class, center its samples using the class mean and calculate the within (class) covariance matrix.
            x_class = x[y==class_label]   
            class_mean = np.mean(x_class, axis=0)
            centered_x_class = x_class-class_mean
            SW += (centered_x_class.T @ centered_x_class)  # rank for every class is min(x_class(linearly independent)-1, d). So after the loop ends SW rank is min(x(linearly independent) - C, d). Then for SW-1 to exist: x(linearly independent) - C >= n_features
            #For every class, center its mean using the overall mean and calculate the between (classes) covariance matrix.
            N_class = x_class.shape[0]
            centered_mean_class = (class_mean - overall_mean).reshape(1, n_features)
            SB += N_class * np.outer(centered_mean_class, centered_mean_class) #Same as centered_mean_class.T @ centered_mean_class but more efficient for rank 1 matrixes. So for every class SB has rank 1 , and after the loop ends the max rank is min(C-1, d)
        print("SW, SB calculated")
        total_lda_matrix = np.linalg.solve(SW, SB)  #No need to store SW-1 in memory that way,
        eigenvalues, eigenvectors = np.linalg.eig(total_lda_matrix)
        sorted_indeces = np.argsort(-eigenvalues)  #Get indeces of eigenvalues in descenting order
        if self.number_of_lda_components==None:
            self.number_of_lda_components = max_rank   #Default n_components value is the max_rank if not set by the user.
        self.needed_eigenvectors = np.real(eigenvectors[:, sorted_indeces[0:self.number_of_lda_components]])  #Matrix of shape (n_features X number_of_lda_components)
        return self

    def transform(self, x_test):
        return np.real(x_test @ self.needed_eigenvectors)  #Transform x_test into shape (n_test X number_of_lda_components)
    
    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)
