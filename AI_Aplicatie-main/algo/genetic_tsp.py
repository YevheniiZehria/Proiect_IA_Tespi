import pygad
import numpy as np
import random
from .tsp_utils import cost_traseu
from .nearest_neighbor import rezolva_tsp_nn

class TSPGenetic:
    def __init__(self, matrice, pop_size=50, n_generatii=200, rata_mutatie=30,
                 selectie="tournament", k_tournament=3, elitism=2):
        self.matrice = matrice
        self.n_orase = len(matrice)
        self.pop_size = pop_size
        self.n_generatii = n_generatii
        self.rata_mutatie = rata_mutatie
        self.selectie = selectie
        self.k_tournament = k_tournament
        self.elitism = elitism

    def fitness_func(self, ga_instance, solutie, solutie_idx):
        """PyGAD maximizează fitness-ul → returnăm negativul distanței."""
        return -cost_traseu(solutie.astype(int).tolist(), self.matrice)

    def ox_crossover(self, parinti, offspring_size, ga_instance):
        """Order Crossover (OX) cu selecție aleatorie a părinților."""
        offspring = []
        
        # Generăm o listă de indecși amestecați pentru a evita încrucișarea secvențială
        indecsi_parinti = list(range(parinti.shape[0]))
        random.shuffle(indecsi_parinti)
        
        for i in range(offspring_size[0]):
            idx1 = indecsi_parinti[i % len(indecsi_parinti)]
            idx2 = indecsi_parinti[(i + 1) % len(indecsi_parinti)]
            
            p1 = parinti[idx1].astype(int).tolist()
            p2 = parinti[idx2].astype(int).tolist()
            n = len(p1)

            cx1, cx2 = sorted(random.sample(range(n), 2))
            copil = [-1] * n
            copil[cx1:cx2 + 1] = p1[cx1:cx2 + 1]

            set_segment = set(copil[cx1:cx2 + 1])
            gene_ramase = [g for g in p2 if g not in set_segment]
            pozitii_libere = [i for i in range(n) if copil[i] == -1]

            for pos, gena in zip(pozitii_libere, gene_ramase):
                copil[pos] = gena

            offspring.append(copil)
            
        return np.array(offspring, dtype=int)


    def inversion_mutation(self, offspring, ga_instance):
        """Inversion Mutation: inversează un sub-traseu pentru a menține muchiile valide."""
        rata = self.rata_mutatie / 100.0
        n = offspring.shape[1]
        
        for i in range(offspring.shape[0]):
            if random.random() < rata:
                # Alegem două puncte de tăietură
                idx1, idx2 = sorted(random.sample(range(n), 2))
                
                # Inversăm segmentul direct în array
                offspring[i][idx1:idx2+1] = offspring[i][idx1:idx2+1][::-1]
                
        return offspring


    def ruleaza(self, ui_callback=None):
        populatie_initiala = []
        
        # INJECTAM elitism empiric: oferim GA-ului traseul NN ca pe o "samanta" puternica
        # nn_traseu, _ = rezolva_tsp_nn(self.n_orase, self.matrice, start=0)
        # populatie_initiala.append(nn_traseu)
        
        for _ in range(self.pop_size):
            perm = list(range(self.n_orase))
            random.shuffle(perm)
            populatie_initiala.append(perm)
        
        # Funcția interioară care oprește algoritmul dacă apeși STOP
        def on_generation_callback(ga_instance):
            best_f = ga_instance.best_solution()[1]
            cst = -best_f if best_f is not None else 0
            
            msg = f"Generația {ga_instance.generations_completed}/{self.n_generatii} - Cost minim: {cst:.2f}"
                
            if ui_callback:
                if ui_callback({'status': msg}):
                    return "stop"
        
        ga_instance = pygad.GA(
            num_generations=self.n_generatii,
            num_parents_mating=max(2, self.pop_size // 2),
            fitness_func=self.fitness_func,
            initial_population=np.array(populatie_initiala, dtype=int),
            crossover_type=self.ox_crossover,
            mutation_type=self.inversion_mutation,
            parent_selection_type=self.selectie,
            K_tournament=self.k_tournament,
            keep_elitism=self.elitism,
            suppress_warnings=True,
            on_generation=on_generation_callback
        )
        ga_instance.run()
        solutie, fitness, _ = ga_instance.best_solution()
        
        # Transformăm fitness-ul negativ înapoi în distanță pozitivă
        cost_real = abs(fitness) 
        
        istoric_convergenta = []
        if hasattr(ga_instance, 'best_solutions_fitness'):
            for gen_idx, f_val in enumerate(ga_instance.best_solutions_fitness):
                istoric_convergenta.append((gen_idx, abs(f_val)))
                
        return solutie.astype(int).tolist(), cost_real, istoric_convergenta