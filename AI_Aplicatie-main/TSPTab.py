import math
import random

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QSpinBox,
    QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox,
    QSplitter, QGraphicsScene, QGraphicsView, QTextEdit,
    QApplication, QMessageBox, QCheckBox
)
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QPolygonF
from PySide6.QtCore import Qt, QPointF, QThread, Signal
import time

# Matplotlib imports for the canvas
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Custom algorithm imports
from algo.nearest_neighbor import rezolva_tsp_nn
from algo.backtracking_tsp import BacktrackingTSP
from algo.simulated_annealing_tsp import rezolva_tsp_sa_custom
from algo.genetic_tsp import TSPGenetic
from algo.hill_climbing_tsp import rezolva_tsp_hc

class AlgorithmWorker(QThread):
    progress_signal = Signal(str)
    finished_signal = Signal(list, float, list, str)
    error_signal = Signal(str)

    def __init__(self, idx, n, dist_matrix, params):
        super().__init__()
        self.idx = idx
        self.n = n
        self.dist_matrix = dist_matrix
        self.params = params
        self._last_print_time = 0

    def check_gui_events(self, status_dict=None):
        if status_dict:
            current_time = time.time()
            if current_time - self._last_print_time > 0.5:
                self.progress_signal.emit(f"{status_dict['status']}")
                self._last_print_time = current_time
        return self.isInterruptionRequested()

    def run(self):
        try:
            nume_algoritm = ""
            traseu, cost, istoric = [], float('inf'), []
            
            if self.idx == 0:
                nume_algoritm = "Nearest Neighbor"
                traseu, cost = rezolva_tsp_nn(self.n, self.dist_matrix, start=0)
                
            elif self.idx == 1:
                mod = self.params['mod']
                nume_algoritm = f"BKT ({mod})"
                bkt = BacktrackingTSP(self.dist_matrix, mod=mod, timp_max=self.params['param'], y_max=int(self.params['param']))
                traseu, cost = bkt.rezolva(ui_callback=self.check_gui_events)
                
            elif self.idx == 2:
                nume_algoritm = "Simulated Annealing"
                kwargs = self.params['kwargs']
                kwargs['ui_callback'] = self.check_gui_events
                traseu, cost, istoric = rezolva_tsp_sa_custom(self.dist_matrix, **kwargs)
                
            elif self.idx == 3:
                nume_algoritm = "Algoritm Genetic"
                ga = TSPGenetic(self.dist_matrix, **self.params['kwargs'])
                traseu, cost, istoric = ga.ruleaza(ui_callback=self.check_gui_events)
                
            elif self.idx == 4:
                nume_algoritm = "Hill Climbing"
                kwargs = self.params['kwargs']
                kwargs['ui_callback'] = self.check_gui_events
                traseu, cost = rezolva_tsp_hc(self.n, self.dist_matrix, **kwargs)

            if not self.isInterruptionRequested():
                self.finished_signal.emit(traseu, cost, istoric, nume_algoritm)
                
        except Exception as e:
            self.error_signal.emit(str(e))


class TSPTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cities = []
        self.dist_matrix = []
        self.stop_requested = False
        self.build_ui()

    def build_ui(self):
        # Layout-ul de bază al ferestrei
        layout = QHBoxLayout(self)
        
        # Panou Stânga: Container pentru Controale și Jurnal
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # ---- Controale Generale ----
        controls_frame = QFrame()
        controls_layout = QVBoxLayout(controls_frame)
        
        controls_layout.addWidget(QLabel("<b>1. Generează Harta</b>"))
        self.spin_cities = QSpinBox()
        self.spin_cities.setRange(2, 999999999)
        self.spin_cities.setValue(15)
        self.spin_cities.setToolTip("Numărul de orașe ce vor fi generate. Valori mari (>15) fac Backtracking-ul foarte lent.")
        
        city_layout = QHBoxLayout()
        city_layout.addWidget(QLabel("Număr orașe:"))
        city_layout.addWidget(self.spin_cities)
        controls_layout.addLayout(city_layout)
        
        btn_gen = QPushButton("Generează Orașe")
        btn_gen.clicked.connect(self.generate_cities)
        controls_layout.addWidget(btn_gen)

        # ---- Selectare Algoritm ----
        controls_layout.addWidget(QLabel("<b>2. Setări Algoritm</b>"))
        self.combo_algo = QComboBox()
        self.combo_algo.addItems([
            "0. Nearest Neighbor",
            "1. Backtracking",
            "2. Simulated Annealing",
            "3. Algoritm Genetic",
            "4. Hill Climbing"
        ])
        self.combo_algo.setToolTip("Selectează algoritmul folosit pentru a găsi cel mai scurt drum.")
        self.combo_algo.currentIndexChanged.connect(self.update_settings_panel)
        controls_layout.addWidget(self.combo_algo)

        from PySide6.QtWidgets import QFormLayout
        # ---- Panouri Setări Dinamice ----
        self.stack_params = QStackedWidget()
        self.stack_params.addWidget(QLabel("Nicio setare necesară (Algoritm constructiv)."))
        
        # Backtracking
        page_bt = QWidget()
        layout_bt = QFormLayout(page_bt)
        self.bt_mod = QComboBox()
        self.bt_mod.addItems(["toate", "prima", "timp", "y_solutii"])
        self.bt_mod.setToolTip("Controlează felul în care se oprește explorarea arborelui.")
        self.bt_mod.currentIndexChanged.connect(self.update_bt_ui)
        layout_bt.addRow("Mod Oprire:", self.bt_mod)
        self.lbl_bt_param = QLabel("Parametru:")
        self.bt_param = QSpinBox()
        self.bt_param.setToolTip("Valoarea de limitare (timp în secunde sau număr de iterații/noduri).")
        layout_bt.addRow(self.lbl_bt_param, self.bt_param)
        self.stack_params.addWidget(page_bt)

        # Simulated Annealing
        page_sa = QWidget()
        layout_sa = QFormLayout(page_sa)
        self.sa_tmax = QDoubleSpinBox(); self.sa_tmax.setRange(1, 999999999); self.sa_tmax.setValue(1000)
        self.sa_tmax.setToolTip("Temperatura inițială. Cu cât este mai mare, cu atât algoritmul acceptă mai ușor rute proaste la început pentru a explora masiv.")
        layout_sa.addRow("Temp. Inițială (T_max):", self.sa_tmax)
        self.sa_init = QComboBox(); self.sa_init.addItems(["random", "nn"])
        self.sa_init.setToolTip("Cum se creează prima rută: Aleatoriu sau folosind Nearest Neighbor ca scut (pornire excelentă).")
        layout_sa.addRow("Stare Inițială:", self.sa_init)
        self.sa_cooling = QComboBox(); self.sa_cooling.addItems(["geometric", "liniara", "logaritmica"])
        self.sa_cooling.setToolTip("Ecuația matematică ce scade temperatura pe măsură ce timpul trece.")
        self.sa_cooling.currentIndexChanged.connect(self.update_sa_ui)
        layout_sa.addRow("Ecuație Răcire:", self.sa_cooling)
        self.lbl_sa_racire = QLabel("Alpha (0.9 - 0.99):")
        self.sa_param_racire = QDoubleSpinBox(); self.sa_param_racire.setDecimals(4); self.sa_param_racire.setRange(0.0001, 999999999); self.sa_param_racire.setValue(0.99)
        self.sa_param_racire.setToolTip("Factorul de răcire. Un 0.99 răcește extrem de lent, permițând multă explorare, pe când 0.5 răcește brusc.")
        layout_sa.addRow(self.lbl_sa_racire, self.sa_param_racire)
        self.sa_stop = QComboBox(); self.sa_stop.addItems(["iteratii", "t_min", "stagnare"])
        self.sa_stop.setToolTip("Alege cum vrei să se declare algoritmul 'Gata': după pași numărați, după ce temp a ajuns la limită, sau când stagnează prea mult.")
        self.sa_stop.currentIndexChanged.connect(self.update_sa_ui)
        layout_sa.addRow("Criteriu Terminare:", self.sa_stop)
        self.lbl_sa_stop = QLabel("Număr Iterații:")
        self.sa_param_stop = QDoubleSpinBox(); self.sa_param_stop.setDecimals(4); self.sa_param_stop.setRange(0.0001, 999999999); self.sa_param_stop.setValue(50000)
        self.sa_param_stop.setToolTip("Valoarea numerică pentru criteriul de oprire setat mai sus.")
        layout_sa.addRow(self.lbl_sa_stop, self.sa_param_stop)
        self.stack_params.addWidget(page_sa)

        # Algoritm Genetic
        page_ga = QWidget()
        layout_ga = QFormLayout(page_ga)
        self.ga_pop = QSpinBox(); self.ga_pop.setRange(2, 999999999); self.ga_pop.setValue(50)
        self.ga_pop.setToolTip("Numărul de soluții candidate (cromozomi) prezente într-o singură generație. Mai multe = diversitate mai mare.")
        self.ga_gen = QSpinBox(); self.ga_gen.setRange(1, 999999999); self.ga_gen.setValue(200)
        self.ga_gen.setToolTip("Numărul de generații pe care se va întinde evoluția (iterațiile algoritmului).")
        self.ga_mut = QSpinBox(); self.ga_mut.setRange(0, 100); self.ga_mut.setValue(30)
        self.ga_mut.setToolTip("Probabilitatea (0-100%) ca o soluție rezultată să sufere o mutație minoră. Ajută la evitarea blocajelor locale.")
        
        self.ga_sel = QComboBox()
        self.ga_sel.addItems(['tournament', 'rws', 'rank', 'sss', 'random'])
        from PySide6.QtCore import Qt
        self.ga_sel.setItemData(0, "Tournament: Se aleg K indivizi și se luptă; cel mai bun devine părinte.", Qt.ItemDataRole.ToolTipRole)
        self.ga_sel.setItemData(1, "RWS (Roulette): Șanse proporționale cu calitatea rutei (ca o roată de noroc).", Qt.ItemDataRole.ToolTipRole)
        self.ga_sel.setItemData(2, "Rank: Selecție bazată pe poziția în clasament (1, 2, 3...), ignorând diferențele imense de scor.", Qt.ItemDataRole.ToolTipRole)
        self.ga_sel.setItemData(3, "SSS (Steady-State): Indivizii de jos sunt înlocuiți imediat de puii celor de sus.", Qt.ItemDataRole.ToolTipRole)
        self.ga_sel.setItemData(4, "Random: Părinții sunt aleși 100% aleator (bun pentru comparații/teste).", Qt.ItemDataRole.ToolTipRole)
        self.ga_sel.setToolTip("Mecanismul prin care se aleg părinții. Extinde lista pentru detalii.")
        self.lbl_ga_k = QLabel("K Turneu:")
        self.ga_k = QSpinBox(); self.ga_k.setRange(1, 100); self.ga_k.setValue(3)
        self.ga_k.setToolTip("Câți indivizi aleatori intră în ringul unui singur turneu de reproducere (folosit doar la selecția Tournament).")
        self.ga_elit = QSpinBox(); self.ga_elit.setRange(0, 999999999); self.ga_elit.setValue(2)
        self.ga_elit.setToolTip("Câți din cei mai performanți indivizi absoluți sunt copiați nemodificați în următoarea generație pentru a nu se pierde aurul.")
        
        layout_ga.addRow("Populație:", self.ga_pop)
        layout_ga.addRow("Generații:", self.ga_gen)
        layout_ga.addRow("Mutație (%):", self.ga_mut)
        layout_ga.addRow("Selecție:", self.ga_sel)
        layout_ga.addRow(self.lbl_ga_k, self.ga_k)
        layout_ga.addRow("Elitism:", self.ga_elit)
        
        self.ga_sel.currentIndexChanged.connect(self.update_ga_ui)
        self.stack_params.addWidget(page_ga)
        
        # Hill Climbing
        page_hc = QWidget()
        layout_hc = QFormLayout(page_hc)
        self.hc_restarts = QSpinBox(); self.hc_restarts.setRange(1, 999999999); self.hc_restarts.setValue(10)
        self.hc_restarts.setToolTip("Numărul de reporniri (random restarts) pe care le face algoritmul pentru a evita să rămână blocat în optimi locali.")
        
        self.hc_init = QComboBox()
        self.hc_init.addItems(['random', 'nn'])
        self.hc_init.setToolTip("Cum se creează starea de start la fiecare restart. 'nn' (Nearest Neighbor) oferă un avantaj imens.")
        
        self.hc_op = QComboBox()
        self.hc_op.addItems(['2-opt', 'swap', 'insert'])
        self.hc_op.setToolTip("Metoda prin care algoritmul generează rutele vecine explorate.")
        from PySide6.QtCore import Qt
        self.hc_op.setItemData(0, "2-opt: Inversează un segment întreg din rută. Ideal pentru a 'descurca' liniile intersectate.", Qt.ItemDataRole.ToolTipRole)
        self.hc_op.setItemData(1, "swap: Interschimbă ordinea a două orașe.", Qt.ItemDataRole.ToolTipRole)
        self.hc_op.setItemData(2, "insert: Mută un oraș într-o cu totul altă poziție din traseu.", Qt.ItemDataRole.ToolTipRole)
        
        layout_hc.addRow("Restarturi:", self.hc_restarts)
        layout_hc.addRow("Stare Inițială:", self.hc_init)
        layout_hc.addRow("Operator Vecini:", self.hc_op)
        self.stack_params.addWidget(page_hc)

        controls_layout.addWidget(self.stack_params)
        self.update_bt_ui()

        # Butoane Execuție / Stop
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Ruleaza")
        self.btn_run.setObjectName("btn_run")
        self.btn_run.clicked.connect(self.run_algorithm)
        
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.clicked.connect(self.request_stop)
        self.btn_stop.setEnabled(False)
        
        btn_layout.addWidget(self.btn_run); btn_layout.addWidget(self.btn_stop)
        controls_layout.addLayout(btn_layout)
        left_layout.addWidget(controls_frame)

        # Jurnal Text în coloana din stânga
        left_layout.addWidget(QLabel("<b>Jurnal Educațional:</b>"))
        self.log_window = QTextEdit(); self.log_window.setReadOnly(True)
        left_layout.addWidget(self.log_window)

        # =========================================================================
        # REFORMARE LAYOUT: Un singur Splitter Orizontal pentru cele 3 elemente mari
        # =========================================================================
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 1. Adăugăm panoul de controale și jurnal (Coloana Stânga)
        self.main_splitter.addWidget(left_panel)
        
        # 2. Adăugăm vizualizarea hărții cu orașele (Coloana Centru)
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.main_splitter.addWidget(self.view)

        # 3. Adăugăm canvas-ul cu graficul de convergență (Coloana Dreapta)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Grafic Convergență (Evoluție Cost)")
        self.ax.set_xlabel("Iterație / Generație")
        self.ax.set_ylabel("Cost Traseu")
        
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.fig.tight_layout()
        
        right_layout.addWidget(self.canvas)
        
        self.btn_save_plot = QPushButton("Salvează Grafic")
        self.btn_save_plot.clicked.connect(self.salveaza_grafic)
        right_layout.addWidget(self.btn_save_plot)
        
        self.main_splitter.addWidget(right_panel)

        # Setăm dimensiunile inițiale estimate (în pixeli) pentru cele 3 secțiuni
        self.main_splitter.setSizes([320, 440, 440])
        
        # Atașăm întregul splitter la layout-ul principal al ferestrei
        layout.addWidget(self.main_splitter)

    # --- Funcții Update UI Dinamic ---
    def update_settings_panel(self):
        self.stack_params.setCurrentIndex(self.combo_algo.currentIndex())

    def update_bt_ui(self):
        mod = self.bt_mod.currentText()
        if mod == 'timp':
            self.lbl_bt_param.setText("Timp maxim (secunde):"); self.bt_param.setRange(1, 999999999); self.bt_param.setValue(5)
            self.lbl_bt_param.show(); self.bt_param.show()
        elif mod == 'y_solutii':
            self.lbl_bt_param.setText("Oprește după X iterații:"); self.bt_param.setRange(1, 999999999); self.bt_param.setValue(2000)
            self.lbl_bt_param.show(); self.bt_param.show()
        else:
            self.lbl_bt_param.hide(); self.bt_param.hide()
            
    def update_ga_ui(self):
        if self.ga_sel.currentText() == "tournament":
            self.lbl_ga_k.show(); self.ga_k.show()
        else:
            self.lbl_ga_k.hide(); self.ga_k.hide()

    def update_sa_ui(self):
        cool = self.sa_cooling.currentText()
        if cool == 'geometric': self.lbl_sa_racire.setText("Alpha (0.9 - 0.99):"); self.sa_param_racire.setValue(0.95)
        elif cool == 'liniara': self.lbl_sa_racire.setText("Scădere Delta:"); self.sa_param_racire.setValue(0.5)
        elif cool == 'logaritmica': self.lbl_sa_racire.setText("Constanta c:"); self.sa_param_racire.setValue(10.0)

        stop = self.sa_stop.currentText()
        if stop == 'iteratii': self.lbl_sa_stop.setText("Număr maxim de iterații:"); self.sa_param_stop.setValue(50000)
        elif stop == 't_min': self.lbl_sa_stop.setText("Temp de îngheț (T_min):"); self.sa_param_stop.setValue(0.01)
        elif stop == 'stagnare': self.lbl_sa_stop.setText("Sistare la X stagnări:"); self.sa_param_stop.setValue(1000)

    def log(self, msj):
        self.log_window.append(msj)
        self.log_window.verticalScrollBar().setValue(self.log_window.verticalScrollBar().maximum())

    def salveaza_grafic(self):
        from PySide6.QtWidgets import QFileDialog
        filepath, _ = QFileDialog.getSaveFileName(self, "Salvează Grafic", "", "Imagini PNG (*.png);;Imagini JPEG (*.jpg)")
        if filepath:
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.log(f"[INFO] Grafic salvat în: {filepath}")

    def generate_cities(self):
        self.scene.clear()
        self.cities = []
        self.log_window.clear()
        
        self.ax.clear()
        self.ax.set_title("Grafic Convergență (Evoluție Cost)")
        self.ax.set_xlabel("Iterație / Generație")
        self.ax.set_ylabel("Cost Traseu")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.canvas.draw()

        n = self.spin_cities.value()
        w, h = self.view.width() - 40, self.view.height() - 40
        
        # Generam orasle logic (intr-un spatiu universal de 1000x1000) 
        # independent de marimea ferestrei pentru corectitudine matematica constanta.
        self.abstract_cities = []
        for i in range(n):
            ax = random.uniform(0, 1000)
            ay = random.uniform(0, 1000)
            self.abstract_cities.append((ax, ay))
            
            # Mapare vizuala pentru afisare pe ecran
            x = (ax / 1000.0) * w + 20
            y = (ay / 1000.0) * h + 20
            self.cities.append((x, y))
            
            self.scene.addEllipse(x - 6, y - 6, 12, 12, QPen(Qt.GlobalColor.black), QBrush(Qt.GlobalColor.darkRed))
            txt = self.scene.addText(str(i)); txt.setPos(x + 5, y - 15)

        # Distantele reale sunt calculate DOAR pe coordonatele abstracte universale
        self.dist_matrix = [[math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) for c2 in self.abstract_cities] for c1 in self.abstract_cities]
        self.log(f"-> Am generat {n} orașe.")


    

    def update_convergence_plot(self, istoric, titlu):
        self.ax.clear()
        
        if istoric:
            # Extragem datele din lista de tupluri: x = iterații/generații, y = costuri
            x_data = [punct[0] for punct in istoric]
            y_data = [punct[1] for punct in istoric]
            
            # Desenăm curba pe grafic
            self.ax.plot(x_data, y_data, label=f"Cost minim", color='#e74c3c', linewidth=2)
            
            # Formatăm axele și titlul pentru estetică
            self.ax.set_title(f"Grafic de Convergență\n({titlu})", fontsize=11, fontweight='bold')
            self.ax.set_xlabel("Iterații / Generații" if "Simulated" in titlu else "Generații", fontsize=10)
            self.ax.set_ylabel("Cost Traseu (Distanță)", fontsize=10)
            self.ax.set_xscale('log')
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.legend(loc="upper right")
            
            # Refacem lățimea secțiunii graficului (aprox. 1/3 din ecran)
            sizes = self.main_splitter.sizes()
            total_width = sum(sizes)
            # Dacă graficul era ascuns (lățime 0), îl redeschidem
            if sizes[2] == 0:
                self.main_splitter.setSizes([int(total_width * 0.3), int(total_width * 0.4), int(total_width * 0.3)])
        else:
            # Stilizare pentru algoritmii care nu au convergență (NN, BKT)
            self.ax.set_title("Grafic de convergență indisponibil\npentru algoritmul selectat", fontsize=10, color='gray')
            self.ax.set_xlabel("")
            self.ax.set_ylabel("")
            self.ax.grid(False)
            
            # Ascundem panoul graficului (setăm lățimea celei de-a treia coloane la 0)
            sizes = self.main_splitter.sizes()
            total_width = sum(sizes)
            self.main_splitter.setSizes([int(total_width * 0.3), int(total_width * 0.7), 0])
            
        try:
            self.fig.tight_layout()
        except Exception:
            pass
            
        self.canvas.draw()

    def draw_path(self, path, cost, nume_algoritm):
        if not path: return
        if path[0] != path[-1]: path.append(path[0])

        self.scene.clear()
        pen_linie = QPen(QColor(41, 128, 185), 2)
        brush_sageata = QBrush(QColor(41, 128, 185))
        
        for i in range(len(path) - 1):
            p1 = self.cities[path[i]]; p2 = self.cities[path[i+1]]
            self.scene.addLine(p1[0], p1[1], p2[0], p2[1], pen_linie)
            
            dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
            angle = math.atan2(dy, dx)
            
            offset = 12
            end_x = p2[0] - offset * math.cos(angle)
            end_y = p2[1] - offset * math.sin(angle)
            
            arrow_size = 12
            arrow_p1 = QPointF(end_x - arrow_size * math.cos(angle - math.pi / 6), end_y - arrow_size * math.sin(angle - math.pi / 6))
            arrow_p2 = QPointF(end_x - arrow_size * math.cos(angle + math.pi / 6), end_y - arrow_size * math.sin(angle + math.pi / 6))
            
            arrow_head = QPolygonF([QPointF(end_x, end_y), arrow_p1, arrow_p2])
            self.scene.addPolygon(arrow_head, QPen(Qt.GlobalColor.transparent), brush_sageata)

        for i, (x, y) in enumerate(self.cities):
            self.scene.addEllipse(x - 6, y - 6, 12, 12, QPen(Qt.GlobalColor.black), QBrush(Qt.GlobalColor.darkRed))
            txt = self.scene.addText(str(i)); txt.setPos(x + 5, y - 15)

        self.log(f"\n[SUCCES] {nume_algoritm} a terminat!")
        self.log(f"Cost total: {cost:.2f}")

    def request_stop(self):
        self.log("\n[!] OPRIRE SOLICITATĂ...")
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.requestInterruption()

    def run_algorithm(self):
        if not self.cities: return
        n = len(self.cities)
        idx = self.combo_algo.currentIndex()
        
        params = {}
        if idx == 1:
            mod = self.bt_mod.currentText(); param = self.bt_param.value()
            if n > 15 and mod == 'toate':
                if QMessageBox.warning(self, "Atenție", "BKT pe 'toate' la N>15 poate bloca aplicația. Continui?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.No: return
            params = {'mod': mod, 'param': param}
            self.log(f"Rulare Backtracking (Mod: {mod})...")
        elif idx == 2:
            t_max = self.sa_tmax.value(); init = self.sa_init.currentText()
            cooling = self.sa_cooling.currentText(); p_racire = self.sa_param_racire.value()
            stop = self.sa_stop.currentText(); p_stop = self.sa_param_stop.value()
            kwargs = {'t_max': t_max, 'mod_init': init, 'cooling': cooling, 'terminare': stop}
            if cooling == 'geometric': kwargs['alpha'] = p_racire
            elif cooling == 'liniara': kwargs['delta'] = p_racire
            elif cooling == 'logaritmica': kwargs['c'] = p_racire
            if stop == 'iteratii': kwargs['steps'] = int(p_stop)
            elif stop == 't_min': kwargs['t_min'] = p_stop
            elif stop == 'stagnare': kwargs['stagnare_max'] = int(p_stop)
            params = {'kwargs': kwargs}
            self.log(f"SA (Răcire: {cooling}, Oprire: {stop})...")
        elif idx == 3:
            pop = self.ga_pop.value(); gen = self.ga_gen.value(); mut = self.ga_mut.value()
            sel = self.ga_sel.currentText(); k = self.ga_k.value(); elit = self.ga_elit.value()
            params = {'kwargs': {'pop_size': pop, 'n_generatii': gen, 'rata_mutatie': mut, 'selectie': sel, 'k_tournament': k, 'elitism': elit}}
            self.log(f"Genetic (Populație: {pop}, Selecție: {sel})...")
        elif idx == 4:
            restarts = self.hc_restarts.value()
            init = self.hc_init.currentText()
            op = self.hc_op.currentText()
            params = {'kwargs': {'restarts': restarts, 'init_type': init, 'operator': op}}
            self.log(f"Hill Climbing ({restarts} restarts, Init: {init}, Op: {op})...")

        self.btn_run.setEnabled(False); self.btn_stop.setEnabled(True)
        self.log("\n" + "="*40)
        
        self.worker = AlgorithmWorker(idx, n, self.dist_matrix, params)
        self.worker.progress_signal.connect(lambda msg: self.log(f"[PROGRES] {msg}"))
        self.worker.error_signal.connect(lambda e: self.log(f"\n[EROARE] {e}"))
        self.worker.finished_signal.connect(self.on_algorithm_finished)
        self.worker.finished.connect(self.on_worker_done)
        self.worker.start()

    def on_algorithm_finished(self, traseu, cost, istoric, nume_algoritm):
        self.draw_path(traseu, cost, nume_algoritm)
        self.update_convergence_plot(istoric, nume_algoritm)

    def on_worker_done(self):
        self.btn_run.setEnabled(True); self.btn_stop.setEnabled(False)
        self.worker.deleteLater()