def rezolva_tsp_nn(n, matrice, start=0):
    """
    Construiește un tur TSP prin euristica celui mai apropiat vecin.

    Args:
        n (int): Numărul de orașe.
        matrice (list of list of float): Matricea de distanțe.
        start (int): Indexul orașului de start.

    Returns:
        tuple: (traseu_optim, cost_total)
    """
    traseu = [start]
    vizitat = {start}
    oras_curent = start
    cost_total = 0.0

    for _ in range(n - 1):
        cel_mai_aproape = -1
        dist_min = float('inf')

        for oras in range(n):
            if oras not in vizitat and matrice[oras_curent][oras] < dist_min:
                dist_min = matrice[oras_curent][oras]
                cel_mai_aproape = oras

        cost_total += dist_min
        traseu.append(cel_mai_aproape)
        vizitat.add(cel_mai_aproape)
        oras_curent = cel_mai_aproape

    # Închidem turul
    cost_total += matrice[oras_curent][start]
    return traseu, cost_total