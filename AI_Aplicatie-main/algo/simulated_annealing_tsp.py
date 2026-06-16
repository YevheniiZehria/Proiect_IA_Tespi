import math
import random
from algo.tsp_utils import cost_traseu
from algo.nearest_neighbor import rezolva_tsp_nn

def rezolva_tsp_sa_custom(matrice, mod_init='random', cooling='geometric', terminare='iteratii',
                          t_max=10000.0, t_min=0.01, steps=50000, alpha=0.99, delta=0.1, c=100.0,
                          stagnare_max=1000, ui_callback=None):
    """
    Implementare educațională pentru Simulated Annealing cu returnare de istoric pentru grafic.
    """
    n = len(matrice)

    # --- 1. Inițializarea stării ---
    if mod_init == 'nn':
        current_tour, _ = rezolva_tsp_nn(n, matrice, start=0)
        if current_tour[0] == current_tour[-1]:
            current_tour = current_tour[:-1]
    else:
        current_tour = list(range(n))
        random.shuffle(current_tour)

    current_cost = cost_traseu(current_tour, matrice)
    best_tour = current_tour.copy()
    best_cost = current_cost

    T = t_max
    iteratie = 1
    stagnare = 0
    
    # --- Colectare date pentru graficul de convergență ---
    istoric_convergenta = []

    while True:
        # Salvăm costul o dată la 100 de iterații pentru performanță grafică
        if iteratie % 100 == 0 or iteratie == 1:
            istoric_convergenta.append((iteratie, best_cost))

        if ui_callback and iteratie % 200 == 0:
            msg = f"Iterația {iteratie} - Temp: {T:.2f} - Cost optim: {best_cost:.2f}"
            if ui_callback({'status': msg}):
                break

        # --- 2. Criterii de terminare ---
        if terminare == 't_min' and T <= t_min:
            break
        elif terminare == 'iteratii' and iteratie >= steps:
            break
        elif terminare == 'stagnare' and stagnare >= stagnare_max:
            break

        # --- 3. Generare vecin & Evaluare O(1) ---
        i, j = sorted(random.sample(range(n), 2))
        
        # Inversarea întregului traseu nu schimbă ciclul, 
        # iar formula dE de mai jos ar da un cost fals negativ.
        if i == 0 and j == n - 1:
            continue
        
        # Nodurile de pe marginile tăieturii
        a = current_tour[i-1]
        b = current_tour[i]
        c = current_tour[j]
        d = current_tour[(j+1) % n]
        
        # Diferența de cost pentru 2-opt (presupunând matrice simetrică)
        dE = (matrice[a][c] + matrice[b][d]) - (matrice[a][b] + matrice[c][d])

        # --- 4. Acceptare ---
        accept = False
        
        if dE < 0:
            accept = True
        else:
            probabilitate = math.exp(-dE / T) if T > 0 else 0
            if random.random() < probabilitate:
                accept = True

        if accept:
            # Inversăm segmentul doar dacă mutarea e acceptată (evităm copieri inutile)
            current_tour[i:j+1] = reversed(current_tour[i:j+1])
            current_cost += dE
            
            # Orice acceptare înseamnă că sistemul este încă "mobil"
            stagnare = 0
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = current_tour.copy()
        else:
            # Stagnăm doar când blocăm o mutare
            stagnare += 1

        # --- 5. Răcirea ---
        if cooling == 'geometric':
            T = alpha * T
        elif cooling == 'liniara':
            T = T - delta
            if T <= 0: T = 1e-10
        elif cooling == 'logaritmica':
            T = c / math.log(1 + iteratie)

        iteratie += 1

    # Adăugăm starea finală în istoric și închidem drumul
    istoric_convergenta.append((iteratie, best_cost))
    best_tour.append(best_tour[0])
    
    return best_tour, best_cost, istoric_convergenta