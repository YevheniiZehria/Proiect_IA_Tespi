import math

def calculeaza_matrice_distante(orase):
    """
    Calculează matricea de distanțe euclidiene între toate perechile de orașe.

    Args:
        orase (list of tuples): Lista de coordonate (x, y) ale orașelor.

    Returns:
        list of list of float: Matricea de distanțe NxN.
    """
    n = len(orase)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = orase[i][0] - orase[j][0]
                dy = orase[i][1] - orase[j][1]
                dist[i][j] = math.sqrt(dx**2 + dy**2)
    return dist

def cost_traseu(traseu, matrice_distante):
    """
    Calculează costul total al unui traseu (ciclu complet).

    Args:
        traseu (list of int): Lista cu indicii orașelor în ordinea vizitării.
        matrice_distante (list of list of float): Matricea NxN.

    Returns:
        float: Costul total al rutei.
    """
    cost = 0.0
    n = len(traseu)
    for i in range(n):
        cost += matrice_distante[traseu[i]][traseu[(i + 1) % n]]
    return cost