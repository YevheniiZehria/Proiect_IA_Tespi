import numpy as np
def main():
    np.random.seed(42)
    mat1 = np.random.randint(1,11,(4,3))
    mat2 = np.random.randint(1,11,(3,5))

    print("Matrix 1:")
    print(mat1)
    print("\nMatrix 2:")
    print(mat2)
    
    matrix_result = mat1 @ mat2
    print("\nThe result of multiply mat1 & mat2:")
    print(matrix_result)

    suma = np.sum(matrix_result)
    print("\nSum of all elements in matrix result:")
    print(suma)

    average_columns = np.average(matrix_result,axis = 0)
    print("\nAverage of each column of matrix result:")
    print(average_columns)

    maxim_matrix = np.max(matrix_result)
    print("\nThe maxim of matrix result:")
    print(maxim_matrix)

    squared_matrix=np.random.randint(1,11,(3,3))
    print("\nSquared matrix:")
    print(squared_matrix)

    squared_matrix_invers=np.linalg.inv(squared_matrix)
    print("\nInverse of squared matrix:") 
    print(squared_matrix_invers)

    determiant=np.linalg.det(squared_matrix)
    print("\nDeterminant of the squared matrix:", determiant)

    multiply=squared_matrix@np.linalg.inv(squared_matrix)
    print("\nProduct of the squared matrix and its inverse:")
    print(multiply)

    print("\nCheck if product is equal to the Identity Matrix:")
    print(np.allclose(multiply, np.eye(3)))

if __name__ == "__main__":
    main()