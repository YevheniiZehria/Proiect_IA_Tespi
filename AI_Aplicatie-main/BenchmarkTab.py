import sys
import random
import math

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QTextEdit, QGraphicsScene, QGraphicsView,
    QSplitter, QFrame, QSpinBox, QComboBox, QStackedWidget, QDoubleSpinBox, 
    QMessageBox, QLineEdit, QFileDialog, QListWidget, QTableWidget, 
    QTableWidgetItem, QCheckBox, QProgressBar, QDialog
)
from BenchmarkSettigns import BKTSettingsDialog, SASettingsDialog, GASettingsDialog, HCSettingsDialog
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QPolygonF
from PySide6.QtCore import Qt, QPointF, QThread, Signal


from algo.nlp_classifier import ClasificatorText
from sklearn.datasets import fetch_20newsgroups

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from algo.nearest_neighbor import rezolva_tsp_nn
from algo.backtracking_tsp import BacktrackingTSP
from algo.simulated_annealing_tsp import rezolva_tsp_sa_custom
from algo.genetic_tsp import TSPGenetic
from algo.hill_climbing_tsp import rezolva_tsp_hc

# Asigură-te că ai importurile necesare la începutul fișierului:
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

import time 

class BenchmarkWorker(QThread):
    progress_signal = Signal(int)
    log_signal = Signal(str)
    plot_update_signal = Signal(dict)
    finished_signal = Signal()

    def __init__(self, parametri_bench, params_algos, checks):
        super().__init__()
        self.p_bench = parametri_bench
        self.p_algos = params_algos
        self.checks = checks
        self._last_print_time = 0

    def check_gui_events(self, status_dict=None):
        if status_dict:
            current_time = time.time()
            if current_time - getattr(self, '_last_print_time', 0) > 0.5:
                # self.log_signal.emit(f"[PROGRES] {status_dict['status']}")
                self._last_print_time = current_time
        return self.isInterruptionRequested()

    def run(self):
        try:
            dimensiuni = self.p_bench['dimensiuni']
            
            if self.p_bench['seed'] is not None:
                random.seed(self.p_bench['seed'])
                self.log_signal.emit(f"Seed fixat la: {self.p_bench['seed']}")
            else:
                random.seed()
                self.log_signal.emit("Seed nefixat (Aleatoriu pur)")
            
            rezultate = {
                'NN': {'x': [], 'timp': [], 'cost': [], 'color': '#3498db'},
                'SA': {'x': [], 'timp': [], 'cost': [], 'color': '#e67e22'},
                'GA': {'x': [], 'timp': [], 'cost': [], 'color': '#2ecc71'},
                'BKT': {'x': [], 'timp': [], 'cost': [], 'color': '#e74c3c'},
                'HC': {'x': [], 'timp': [], 'cost': [], 'color': '#9b59b6'}
            }
            
            self.log_signal.emit("--- Începere Benchmark ---")
            
            for idx, n in enumerate(dimensiuni):
                if self.isInterruptionRequested(): break
                self.log_signal.emit(f"\nGenerare hartă cu N={n}...")
                
                orase = [(random.uniform(0, 1000), random.uniform(0, 1000)) for _ in range(n)]
                dist_matrix = [[math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) for c2 in orase] for c1 in orase]
                
                # 1. NN
                if self.checks['nn'] and not self.isInterruptionRequested():
                    start_time = time.perf_counter()
                    traseu, cost = rezolva_tsp_nn(n, dist_matrix, start=0)
                    durata = time.perf_counter() - start_time
                    rezultate['NN']['x'].append(n); rezultate['NN']['timp'].append(durata); rezultate['NN']['cost'].append(cost)
                
                # 2. SA
                if self.checks['sa'] and not self.isInterruptionRequested():
                    start_time = time.perf_counter()
                    kwargs = dict(self.p_algos['sa'])
                    kwargs['ui_callback'] = self.check_gui_events
                    traseu, cost, _ = rezolva_tsp_sa_custom(dist_matrix, **kwargs)
                    durata = time.perf_counter() - start_time
                    rezultate['SA']['x'].append(n); rezultate['SA']['timp'].append(durata); rezultate['SA']['cost'].append(cost)
                    
                # 3. GA
                if self.checks['ga'] and not self.isInterruptionRequested():
                    start_time = time.perf_counter()
                    ga = TSPGenetic(dist_matrix, **self.p_algos['ga'])
                    traseu, cost, _ = ga.ruleaza(ui_callback=self.check_gui_events)
                    durata = time.perf_counter() - start_time
                    rezultate['GA']['x'].append(n); rezultate['GA']['timp'].append(durata); rezultate['GA']['cost'].append(cost)

                # 4. BKT
                if self.checks['bkt'] and not self.isInterruptionRequested():
                    if n <= 12 or self.p_algos['bkt']['mod'] != 'toate':
                        start_time = time.perf_counter()
                        bkt = BacktrackingTSP(dist_matrix, **self.p_algos['bkt'])
                        traseu, cost = bkt.rezolva(ui_callback=self.check_gui_events)
                        durata = time.perf_counter() - start_time
                        rezultate['BKT']['x'].append(n); rezultate['BKT']['timp'].append(durata); rezultate['BKT']['cost'].append(cost)
                    else:
                        self.log_signal.emit(f"BKT sărit pentru N={n} (modul 'toate' prea lent).")
                        
                # 5. HC
                if self.checks['hc'] and not self.isInterruptionRequested():
                    start_time = time.perf_counter()
                    kwargs = dict(self.p_algos['hc'])
                    kwargs['ui_callback'] = self.check_gui_events
                    traseu, cost = rezolva_tsp_hc(n, dist_matrix, **kwargs)
                    durata = time.perf_counter() - start_time
                    rezultate['HC']['x'].append(n); rezultate['HC']['timp'].append(durata); rezultate['HC']['cost'].append(cost)
                    
                self.progress_signal.emit(idx + 1)
                self.plot_update_signal.emit(rezultate)
                
            if self.isInterruptionRequested():
                self.log_signal.emit("\n--- Benchmark Oprit de Utilizator ---")
            else:
                self.log_signal.emit("\n--- Benchmark Finalizat ---")
                
        except Exception as e:
            self.log_signal.emit(f"\n[EROARE] Benchmark: {e}")
            
        finally:
            self.finished_signal.emit()

class BenchmarkTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # Inițializăm dictionarele cu parametrii impliciți (identici cu cei din TSPTab)
        self.params_bkt = {'mod': 'toate', 'param': 5}
        self.params_sa = {
            't_max': 1000.0, 'mod_init': 'random', 'cooling': 'geometric', 
            'p_racire': 0.95, 'terminare': 'iteratii', 'p_stop': 5000.0
        }
        self.params_ga = {'pop_size': 50, 'n_generatii': 200, 'rata_mutatie': 30, 'selectie': 'tournament', 'k_tournament': 3, 'elitism': 2}
        self.params_hc = {'restarts': 10, 'init_type': 'random', 'operator': '2-opt'}
        
        self.build_ui()

    def build_ui(self):
        layout = QHBoxLayout(self)

        # PANOU STÂNGA - Controale
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        controls_frame = QFrame()
        controls_layout = QVBoxLayout(controls_frame)
        
        controls_layout.addWidget(QLabel("<h2>Setări Benchmark TSP</h2>"))
        controls_layout.addWidget(QLabel("Testează performanța algoritmilor pe măsură ce N crește."))
        
        # Setări N
        grid_n = QHBoxLayout()
        self.spin_n_min = QSpinBox(); self.spin_n_min.setRange(2, 999999999); self.spin_n_min.setValue(5)
        self.spin_n_min.setToolTip("Numărul minim de orașe de la care pornește benchmark-ul.")
        self.spin_n_max = QSpinBox(); self.spin_n_max.setRange(2, 999999999); self.spin_n_max.setValue(50)
        self.spin_n_max.setToolTip("Numărul maxim de orașe la care se oprește benchmark-ul.")
        self.spin_n_step = QSpinBox(); self.spin_n_step.setRange(1, 999999999); self.spin_n_step.setValue(5)
        self.spin_n_step.setToolTip("Pasul cu care crește numărul de orașe (ex: 5, 10, 15...).")
        
        grid_n.addWidget(QLabel("N Min:")); grid_n.addWidget(self.spin_n_min)
        grid_n.addWidget(QLabel("N Max:")); grid_n.addWidget(self.spin_n_max)
        grid_n.addWidget(QLabel("Pas:")); grid_n.addWidget(self.spin_n_step)
        controls_layout.addLayout(grid_n)

        # Setare Seed
        grid_seed = QHBoxLayout()
        self.input_seed = QLineEdit()
        self.input_seed.setPlaceholderText("Seed (Ex: 42) - Lasă gol pt. random")
        grid_seed.addWidget(QLabel("Seed Aleator (Opțional):"))
        grid_seed.addWidget(self.input_seed)
        controls_layout.addLayout(grid_seed)
        

        # Selecție Algoritm + Butoane Setări inline
        controls_layout.addWidget(QLabel("<b>Algoritmi de testat:</b>"))
        
        # NN Row
        row_nn = QHBoxLayout()
        self.chk_nn = QCheckBox("Nearest Neighbor")
        self.chk_nn.setChecked(True)
        row_nn.addWidget(self.chk_nn)
        row_nn.addStretch()
        controls_layout.addLayout(row_nn)
        
        # SA Row
        row_sa = QHBoxLayout()
        self.chk_sa = QCheckBox("Simulated Annealing")
        self.chk_sa.setChecked(True)
        self.btn_set_sa = QPushButton("Setări")
        self.btn_set_sa.setFixedWidth(50)
        self.btn_set_sa.setToolTip("Setări avansate Simulated Annealing")
        self.btn_set_sa.clicked.connect(self.open_sa_settings)
        row_sa.addWidget(self.chk_sa)
        row_sa.addWidget(self.btn_set_sa)
        controls_layout.addLayout(row_sa)
        
        # GA Row
        row_ga = QHBoxLayout()
        self.chk_ga = QCheckBox("Algoritm Genetic")
        self.chk_ga.setChecked(True)
        self.btn_set_ga = QPushButton("Setări")
        self.btn_set_ga.setFixedWidth(50)
        self.btn_set_ga.setToolTip("Setări avansate Algoritm Genetic")
        self.btn_set_ga.clicked.connect(self.open_ga_settings)
        row_ga.addWidget(self.chk_ga)
        row_ga.addWidget(self.btn_set_ga)
        controls_layout.addLayout(row_ga)
        
        # BKT Row
        row_bkt = QHBoxLayout()
        self.chk_bkt = QCheckBox("Backtracking (Limitat la N<=12)")
        self.btn_set_bkt = QPushButton("Setări")
        self.btn_set_bkt.setFixedWidth(50)
        self.btn_set_bkt.setToolTip("Setări avansate Backtracking")
        self.btn_set_bkt.clicked.connect(self.open_bkt_settings)
        row_bkt.addWidget(self.chk_bkt)
        row_bkt.addWidget(self.btn_set_bkt)
        controls_layout.addLayout(row_bkt)

        # HC Row
        row_hc = QHBoxLayout()
        self.chk_hc = QCheckBox("Hill Climbing")
        self.chk_hc.setChecked(True)
        self.btn_set_hc = QPushButton("Setări")
        self.btn_set_hc.setFixedWidth(50)
        self.btn_set_hc.setToolTip("Setări avansate Hill Climbing")
        self.btn_set_hc.clicked.connect(self.open_hc_settings)
        row_hc.addWidget(self.chk_hc)
        row_hc.addWidget(self.btn_set_hc)
        controls_layout.addLayout(row_hc)

        # Butoane Execuție și Oprire
        run_stop_layout = QHBoxLayout()
        self.btn_run_bench = QPushButton("Pornește Benchmark")
        self.btn_run_bench.setObjectName("btn_run")
        self.btn_run_bench.clicked.connect(self.run_benchmark)
        
        self.btn_stop_bench = QPushButton("Stop")
        self.btn_stop_bench.setObjectName("btn_stop")
        self.btn_stop_bench.clicked.connect(self.request_stop)
        self.btn_stop_bench.setEnabled(False)
        
        run_stop_layout.addWidget(self.btn_run_bench)
        run_stop_layout.addWidget(self.btn_stop_bench)
        controls_layout.addLayout(run_stop_layout)
        
        self.progress = QProgressBar()
        self.progress.setValue(0)
        controls_layout.addWidget(self.progress)
        
        self.log_bench = QTextEdit()
        self.log_bench.setReadOnly(True)
        controls_layout.addWidget(self.log_bench)
        
        left_layout.addWidget(controls_frame)

        # PANOU DREAPTA - Grafice
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.fig = Figure(figsize=(6, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        
        self.ax_timp = self.fig.add_subplot(211)
        self.ax_cost = self.fig.add_subplot(212)
        
        self.setup_axes()
        right_layout.addWidget(self.canvas)

        self.btn_save_plot = QPushButton("Salvează Grafic")
        self.btn_save_plot.clicked.connect(self.salveaza_grafic)
        right_layout.addWidget(self.btn_save_plot)

        # Splitter principal
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 650])
        layout.addWidget(splitter)
        
        # Variabila pt starea de oprire
        self.stop_requested = False
        self.last_gui_update = time.time()

    # --- Metode deschidere dialoguri modale ---
    def open_bkt_settings(self):
        dialog = BKTSettingsDialog(self.params_bkt, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.params_bkt = dialog.get_params()
            self.log(f"[INFO] Setări BKT salvate: {self.params_bkt}")

    def open_sa_settings(self):
        dialog = SASettingsDialog(self.params_sa, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.params_sa = dialog.get_params()
            self.log(f"[INFO] Setări SA salvate: {self.params_sa}")

    def open_ga_settings(self):
        dialog = GASettingsDialog(self.params_ga, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.params_ga = dialog.get_params()
            self.log(f"[INFO] Setări GA salvate: {self.params_ga}")

    def open_hc_settings(self):
        dialog = HCSettingsDialog(self.params_hc, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.params_hc = dialog.get_params()
            self.log(f"[INFO] Setări HC salvate: {self.params_hc}")

    def setup_axes(self):
        self.ax_timp.clear()
        self.ax_timp.set_title("Timp de Execuție vs Complexitate (N)")
        self.ax_timp.set_xlabel("Număr Orașe (N)")
        self.ax_timp.set_ylabel("Timp (secunde) - Log Scale")
        self.ax_timp.set_yscale('log')
        self.ax_timp.grid(True, linestyle='--', alpha=0.5)

        self.ax_cost.clear()
        self.ax_cost.set_title("Calitatea Soluției (Cost) vs Complexitate (N)")
        self.ax_cost.set_xlabel("Număr Orașe (N)")
        self.ax_cost.set_ylabel("Cost Traseu (mai mic e mai bun)")
        self.ax_cost.grid(True, linestyle='--', alpha=0.5)

    def log(self, text):
        self.log_bench.append(text)

    def request_stop(self):
        self.stop_requested = True
        self.log("\n[!] OPRIRE SOLICITATĂ... Aștept finalizarea operațiunii curente.")
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.requestInterruption()

    def run_benchmark(self):
        n_min = self.spin_n_min.value()
        n_max = self.spin_n_max.value()
        pas = self.spin_n_step.value()
        
        dimensiuni = list(range(n_min, n_max + 1, pas))
        if not dimensiuni: return
        
        seed_val = self.input_seed.text().strip()
        seed_int = int(seed_val) if seed_val.isdigit() else None
        
        self.btn_run_bench.setEnabled(False)
        self.btn_stop_bench.setEnabled(True)
        self.stop_requested = False
        
        self.progress.setMaximum(len(dimensiuni))
        self.progress.setValue(0)
        self.log_bench.clear()
        self.setup_axes()
        self.canvas.draw()
        
        p_bench = {'dimensiuni': dimensiuni, 'seed': seed_int}
        checks = {
            'nn': self.chk_nn.isChecked(),
            'sa': self.chk_sa.isChecked(),
            'ga': self.chk_ga.isChecked(),
            'bkt': self.chk_bkt.isChecked(),
            'hc': self.chk_hc.isChecked()
        }
        
        kwargs_sa = {
            't_max': self.params_sa['t_max'],
            'mod_init': self.params_sa['mod_init'],
            'cooling': self.params_sa['cooling'],
            'terminare': self.params_sa['terminare']
        }
        cool = self.params_sa['cooling']
        p_racire = self.params_sa['p_racire']
        if cool == 'geometric': kwargs_sa['alpha'] = p_racire
        elif cool == 'liniara': kwargs_sa['delta'] = p_racire
        elif cool == 'logaritmica': kwargs_sa['c'] = p_racire
        stop = self.params_sa['terminare']
        p_stop = self.params_sa['p_stop']
        if stop == 'iteratii': kwargs_sa['steps'] = int(p_stop)
        elif stop == 't_min': kwargs_sa['t_min'] = p_stop
        elif stop == 'stagnare': kwargs_sa['stagnare_max'] = int(p_stop)

        p_algos = {
            'bkt': {'mod': self.params_bkt['mod'], 'timp_max': self.params_bkt['param'], 'y_max': int(self.params_bkt['param'])},
            'sa': kwargs_sa,
            'ga': {
                'pop_size': self.params_ga['pop_size'], 'n_generatii': self.params_ga['n_generatii'],
                'rata_mutatie': self.params_ga['rata_mutatie'], 'selectie': self.params_ga['selectie'],
                'k_tournament': self.params_ga['k_tournament'], 'elitism': self.params_ga['elitism']
            },
            'hc': {
                'restarts': self.params_hc['restarts'], 
                'init_type': self.params_hc.get('init_type', 'random'), 
                'operator': self.params_hc.get('operator', '2-opt')
            }
        }
        
        self.worker = BenchmarkWorker(p_bench, p_algos, checks)
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.plot_update_signal.connect(self.update_plots)
        self.worker.finished_signal.connect(self.on_benchmark_finished)
        self.worker.start()

    def on_benchmark_finished(self):
        self.btn_run_bench.setEnabled(True)
        self.btn_stop_bench.setEnabled(False)
        self.worker.deleteLater()

    def update_plots(self, rezultate):
        self.setup_axes()
        
        culori = {'NN': 'cyan', 'BKT': 'red', 'SA': 'yellow', 'GA': 'magenta', 'HC': 'green'}
        markere = {'NN': 'o', 'BKT': 's', 'SA': '^', 'GA': 'v', 'HC': 'x'}
        
        for algo, date in rezultate.items():
            if not date['x']: continue
            self.ax_timp.plot(date['x'], date['timp'], label=algo, color=culori.get(algo, 'white'), marker=markere.get(algo, 'o'), markersize=4)
            self.ax_cost.plot(date['x'], date['cost'], label=algo, color=culori.get(algo, 'white'), marker=markere.get(algo, 'o'), markersize=4)
            
        self.ax_timp.legend(loc='upper left', fontsize=8)
        self.ax_cost.legend(loc='upper right', fontsize=8)
        
        self.fig.tight_layout()
        self.canvas.draw()

    def salveaza_grafic(self):
        from PySide6.QtWidgets import QFileDialog
        filepath, _ = QFileDialog.getSaveFileName(self, "Salvează Grafic Benchmark", "", "Imagini PNG (*.png);;Imagini JPEG (*.jpg)")
        if filepath:
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
            self.log(f"[INFO] Grafic salvat în: {filepath}")
        QApplication.processEvents()