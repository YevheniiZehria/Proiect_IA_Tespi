import numpy as np
def main():
    np.random.seed(42)  
    mat1=np.random.randint(1,11,(4,3))
    mat2=np.random.randint(1,11,(3,5))
    
    print("Matricea 1:")
    print(mat1) 
    print("\nMatricea 2:")
    print(mat2)
    
    matrice_resultat=mat1 @ mat2
    print("\nRezultatul înmulțirii matricelor:")
    print(matrice_resultat)
    
    suma= np.sum(matrice_resultat)
    print("\nSuma tuturor elementelor din matricea rezultată:", suma)
    medie_coloane=np.average(matrice_resultat,axis=0)
    print("\nMedia elementelor din fiecare coloană a matricei rezultante:")
    print(medie_coloane)
    maxim_matrice=np.max(matrice_resultat)
    print("\nValoarea maximă din matricea rezultată:", maxim_matrice)
    matrice_patratica=np.random.randint(1,11,(3,3))
    print("\nMatricea patratică:")
    print(matrice_patratica)
    matrice_patratica_inversa=np.linalg.inv(matrice_patratica)
    print("\nInversa matricei patratice:") 
    print(matrice_patratica_inversa)
    determiant=np.linalg.det(matrice_patratica)
    print("\nDeterminantul matricei patratice:", determiant)
    produs=matrice_patratica@np.linalg.inv(matrice_patratica)
    print("\nProdusul matricei patratice și a inversului său:")
    print(produs)
    print("\nVerificare dacă produsul este egal cu matricea identitate:")
    print(np.allclose(produs, np.eye(3)))
if __name__ == "__main__":
    main()