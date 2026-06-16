import time
from .nearest_neighbor import rezolva_tsp_nn

class BacktrackingTSP:
    def __init__(self, matrice, mod='toate', timp_max=10, y_max=10):
        self.matrice = matrice
        self.n = len(matrice)
        self.mod = mod          # 'prima', 'toate', 'timp', 'y_solutii'
        self.timp_max = timp_max
        self.y_max = y_max
        
        # Initializam bound-ul superior cu Nearest Neighbor doar pentru explorarile exhaustive/timp.
        # Pentru 'prima' si 'y_solutii' vrem sa lasam algoritmul sa gaseasca solutii fara sa le taie.
        if self.mod in ['toate', 'timp']:
            nn_traseu, nn_cost = rezolva_tsp_nn(self.n, self.matrice, start=0)
            self.cost_minim = nn_cost
            self.traseu_optim = nn_traseu
        else:
            self.cost_minim = float('inf')
            self.traseu_optim = []
        self.nr_solutii = 0
        self.iterații = 0
        self.oprire = False
        self.timp_start = 0

    def check_stop(self, ui_callback, msg="Explorare în curs..."):
        # Permitem interfeței să respire și preluăm statusul de "Stop"
        if ui_callback and ui_callback({'status': msg}):
            self.oprire = True
            
        # 2. Oprire la expirarea timpului
        if self.mod == 'timp' and (time.time() - self.timp_start) >= self.timp_max:
            self.oprire = True
            
        return self.oprire

    def rezolva(self, start=0, ui_callback=None):
        self.timp_start = time.time()
        vizitat = [False] * self.n
        vizitat[start] = True
        self._bkt(start, vizitat, [start], 0.0, ui_callback)
        return self.traseu_optim, self.cost_minim

    def _bkt(self, oras_curent, vizitat, traseu, cost, ui_callback):
        # Fast fail if a stop signal was already caught in a deeper or parallel branch
        if self.oprire:
            return

        # Count recursive iterations for UI updates
        self.iterații += 1 
        
        # Only poll the UI and check time constraints periodically
        if self.iterații % 2000 == 0:
            if self.check_stop(ui_callback, f"Iterația {self.iterații} - Explorat: {len(traseu)}/{self.n}"):
                return

        # Am găsit un traseu complet
        if len(traseu) == self.n:
            cost_total = cost + self.matrice[oras_curent][traseu[0]]
            
            if cost_total < self.cost_minim:
                self.cost_minim = cost_total
                self.traseu_optim = traseu.copy()
            
            # Now we correctly count actual complete solutions found
            self.nr_solutii += 1 
            
            # Condiție de oprire pe prima soluție completă găsită
            if self.mod == 'prima':
                self.oprire = True
            # Condiție de oprire pe un număr fix de soluții completate
            elif self.mod == 'y_solutii' and self.nr_solutii >= self.y_max:
                self.oprire = True
                
            return

        # Explorare
        for urmator in range(self.n):
            # Check the boolean directly—much faster than a function call
            if self.oprire:
                return
                
            if vizitat[urmator]:
                continue
                
            cost_nou = cost + self.matrice[oras_curent][urmator]
            
            # Pruning (Tăierea ramurilor ineficiente) aplicat doar la explorarile exhaustive
            if self.mod in ['toate', 'timp'] and cost_nou >= self.cost_minim:
                continue

            vizitat[urmator] = True
            traseu.append(urmator)
            
            self._bkt(urmator, vizitat, traseu, cost_nou, ui_callback)
            
            traseu.pop()
            vizitat[urmator] = False