import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 


if __name__ == "__main__":
    np.random.seed(100)
    arrA = np.random.randint( 1,10,size=(4,3))
    print(arrA)
    arrB = np.random.randint(1,10,size=(3,5))
    print(arrB)
    Product = arrA @ arrB
    print(Product)

    print("Product sum " , np.sum(Product))
    print("Product Media ",np.mean(Product,axis=0))
    print("Product Max ", np.max(Product))

    square1 = np.random.randint(1,10,size=(3,3))
    invsqure1 = np.linalg.inv(square1)
    det = np.linalg.det(square1)
    print(square1)
    print(invsqure1)
    print(det)

    print(square1 @ invsqure1)