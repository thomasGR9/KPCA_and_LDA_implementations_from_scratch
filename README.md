# **Kernel PCA & LDA from Scratch**

This repository contains a **from-scratch implementation** of two powerful dimensionality reduction techniques:

- **Kernel Principal Component Analysis (Kernel PCA)**
- **Linear Discriminant Analysis (LDA)**

---

## **Features**

### **Kernel PCA**
- **Multiple Kernel Support**:  
  - **Linear**, **Polynomial**, and **RBF (Radial Basis Function)**
- **Efficient Kernel Computation**:  
  - Utilizes **chunk-based processing** and **block-wise operations** to compute the RBF kernel, reducing peak memory consumption.
  - Employs **vectorized operations** with NumPy for fast pairwise computations.
- **Memory Management Options**:  
  - **MemoryProblem Flag**: Alters the centering process to handle large kernel matrices more efficiently.
  - **Disk-Saving Option**: Saves intermediate kernel matrices as `.joblib` files on disk to prevent memory overload.

### **LDA**
- **From-Scratch LDA**:  
  - Computes the within-class and between-class scatter matrices.
  - Solves the generalized eigenvalue problem efficiently to obtain the discriminant components.
- **Robustness**:  
  - Validates component limits based on the number of classes and feature dimensions.
