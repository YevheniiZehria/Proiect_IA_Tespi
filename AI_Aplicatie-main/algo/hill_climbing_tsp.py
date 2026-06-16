from .tsp_utils import cost_traseu
from algo.nearest_neighbor import rezolva_tsp_nn
import random

class TSPHillClimbing:
    def __init__(self, initial_state, matrice_distante, operator='2-opt'):
        self.initial_state = tuple(initial_state)
        self.matrice = matrice_distante
        self.n = len(initial_state)
        self.operator = operator

    def actions(self, state):
        """Generează vecinii folosind operatorul specificat."""
        actiuni = []
        if self.operator == '2-opt':
            for i in range(self.n - 1):
                for j in range(i + 1, self.n):
                    actiuni.append(('2-opt', i, j))
        elif self.operator == 'swap':
            for i in range(self.n - 1):
                for j in range(i + 1, self.n):
                    actiuni.append(('swap', i, j))
        elif self.operator == 'insert':
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        actiuni.append(('insert', i, j))
        return actiuni

    def result(self, state, action):
        """Aplică mișcarea selectată pe starea curentă."""
        op, i, j = action
        lista = list(state)
        
        if op == '2-opt':
            lista[i:j+1] = reversed(lista[i:j+1])
        elif op == 'swap':
            lista[i], lista[j] = lista[j], lista[i]
        elif op == 'insert':
            oras = lista.pop(i)
            lista.insert(j, oras)
            
        return tuple(lista)

    def value(self, state):
        """
        Funcția de evaluare (simpleai maximizează).
        Returnăm negativul costului pentru a minimiza distanța.
        """
        return -cost_traseu(list(state), self.matrice)

def rezolva_tsp_hc(n, matrice, restarts=10, init_type='random', operator='2-opt', ui_callback=None):
    """Rulează Hill Climbing complet manual pentru a suporta GUI events (evită freeze-ul aplicației)."""
    best_traseu = list(range(n))
    best_cost = float('inf')
    
    for r in range(restarts):
        if init_type == 'random':
            orase = list(range(n))
            random.shuffle(orase)
        elif init_type == 'nn':
            orase, _ = rezolva_tsp_nn(n, matrice, start=random.randint(0, n-1))
            
        problema = TSPHillClimbing(orase, matrice, operator=operator)
        
        current_state = problema.initial_state
        current_value = problema.value(current_state)
        
        # Inner loop (pășește pe deal)
        while True:
            # Apelăm UI callback la fiecare pas pentru a nu îngheța interfața și a detecta butonul STOP
            if ui_callback:
                stop = ui_callback({'status': f"Restart {r+1}/{restarts} | Cost minim curent: {best_cost if best_cost != float('inf') else 0:.2f}"})
                if stop:
                    return best_traseu, best_cost
                    
            actiuni = problema.actions(current_state)
            best_neighbor = None
            best_neighbor_value = -float('inf')
            
            # Evaluăm vecinii
            for idx, act in enumerate(actiuni):
                neighbor = problema.result(current_state, act)
                val = problema.value(neighbor)
                if val > best_neighbor_value:
                    best_neighbor = neighbor
                    best_neighbor_value = val
                    
                # Permitem interfeței să respire la fiecare 100 de evaluări
                if idx % 100 == 0 and ui_callback:
                    if ui_callback():
                        return best_traseu, best_cost
                    
            # Dacă nu am găsit o mișcare care să îmbunătățească STRICT starea, suntem într-un optim local
            if best_neighbor_value <= current_value:
                break
                
            current_state = best_neighbor
            current_value = best_neighbor_value
            
        # S-a terminat un restart, salvăm dacă e cel mai bun global
        cost = -current_value
        if cost < best_cost:
            best_cost = cost
            best_traseu = list(current_state)
            
    return best_traseu, best_cost