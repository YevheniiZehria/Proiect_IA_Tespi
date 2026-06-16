import sys
import random
import math
import os 
import json
import time 

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QTextEdit, QGraphicsScene, QGraphicsView,
    QSplitter, QFrame, QSpinBox, QComboBox, QStackedWidget, QDoubleSpinBox, 
    QMessageBox, QLineEdit, QFileDialog, QListWidget, QTableWidget, 
    QTableWidgetItem, QHeaderView, QInputDialog, QCheckBox, QProgressBar,
    QDialog, QDialogButtonBox
)
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QPolygonF
from PySide6.QtCore import Qt, QPointF

# --- Importuri Canvas Matplotlib pentru Embed în PySide6 ---
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Algoritmi
from algo.nearest_neighbor import rezolva_tsp_nn
from algo.backtracking_tsp import BacktrackingTSP
from algo.simulated_annealing_tsp import rezolva_tsp_sa_custom
from algo.genetic_tsp import TSPGenetic

from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


# =========================================================
# CLASE DIALOG PENTRU SETĂRI AVANSATE (Preluat din TSPTab)
# =========================================================

from PySide6.QtWidgets import QFormLayout

class BKTSettingsDialog(QDialog):
    def __init__(self, current_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setări Avansate Backtracking")
        self.setMinimumWidth(280)
        layout = QFormLayout(self)
        
        self.bt_mod = QComboBox()
        self.bt_mod.addItems(["toate", "prima", "timp", "y_solutii"])
        self.bt_mod.setCurrentText(current_params['mod'])
        self.bt_mod.setToolTip("Controlează felul în care se oprește explorarea arborelui.")
        self.bt_mod.currentIndexChanged.connect(self.update_bt_ui)
        layout.addRow("Mod Oprire:", self.bt_mod)
        
        self.lbl_bt_param = QLabel("Parametru:")
        self.bt_param = QSpinBox()
        self.bt_param.setRange(1, 999999999)
        self.bt_param.setValue(current_params['param'])
        self.bt_param.setToolTip("Valoarea de limitare (timp în secunde sau număr de iterații/noduri).")
        layout.addRow(self.lbl_bt_param, self.bt_param)
        
        self.update_bt_ui()
        
        buttons = QDialogButtonBox()
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")
        buttons.addButton(btn_ok, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(btn_cancel, QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
   
        
    def update_bt_ui(self):
        mod = self.bt_mod.currentText()
        if mod == 'timp':
            self.lbl_bt_param.setText("Timp maxim (secunde):")
            self.bt_param.setRange(1, 999999999)
            self.lbl_bt_param.show(); self.bt_param.show()
        elif mod == 'y_solutii':
            self.lbl_bt_param.setText("Oprește după X iterații:")
            self.bt_param.setRange(1, 999999999)
            self.lbl_bt_param.show(); self.bt_param.show()
        else:
            self.lbl_bt_param.hide(); self.bt_param.hide()
            
    def get_params(self):
        return {'mod': self.bt_mod.currentText(), 'param': self.bt_param.value()}


class SASettingsDialog(QDialog):
    def __init__(self, current_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setări Avansate Simulated Annealing")
        self.setMinimumWidth(320)
        layout = QFormLayout(self)
        
        self.sa_tmax = QDoubleSpinBox(); self.sa_tmax.setRange(1, 999999999); self.sa_tmax.setValue(current_params['t_max'])
        self.sa_tmax.setToolTip("Temperatura inițială. Cu cât este mai mare, cu atât algoritmul acceptă mai ușor rute proaste la început pentru a explora masiv.")
        layout.addRow("Temp. Inițială (T_max):", self.sa_tmax)
        
        self.sa_init = QComboBox(); self.sa_init.addItems(["random", "nn"])
        self.sa_init.setCurrentText(current_params['mod_init'])
        self.sa_init.setToolTip("Cum se creează prima rută: Aleatoriu sau folosind Nearest Neighbor ca scut (pornire excelentă).")
        layout.addRow("Stare Inițială:", self.sa_init)
        
        self.sa_cooling = QComboBox(); self.sa_cooling.addItems(["geometric", "liniara", "logaritmica"])
        self.sa_cooling.setCurrentText(current_params['cooling'])
        self.sa_cooling.setToolTip("Ecuația matematică ce scade temperatura pe măsură ce timpul trece.")
        self.sa_cooling.currentIndexChanged.connect(self.update_sa_ui)
        layout.addRow("Ecuație Răcire:", self.sa_cooling)
        
        self.lbl_sa_racire = QLabel("Alpha (0.9 - 0.99):")
        self.sa_param_racire = QDoubleSpinBox(); self.sa_param_racire.setDecimals(4); self.sa_param_racire.setRange(0.0001, 999999999); self.sa_param_racire.setValue(current_params['p_racire'])
        self.sa_param_racire.setToolTip("Factorul de răcire. Un 0.99 răcește extrem de lent, permițând multă explorare, pe când 0.5 răcește brusc.")
        layout.addRow(self.lbl_sa_racire, self.sa_param_racire)
        
        self.sa_stop = QComboBox(); self.sa_stop.addItems(["iteratii", "t_min", "stagnare"])
        self.sa_stop.setCurrentText(current_params['terminare'])
        self.sa_stop.setToolTip("Alege cum vrei să se declare algoritmul 'Gata': după pași numărați, după ce temp a ajuns la limită, sau când stagnează prea mult.")
        self.sa_stop.currentIndexChanged.connect(self.update_sa_ui)
        layout.addRow("Criteriu Terminare:", self.sa_stop)
        
        self.lbl_sa_stop = QLabel("Număr Iterații:")
        self.sa_param_stop = QDoubleSpinBox(); self.sa_param_stop.setDecimals(4); self.sa_param_stop.setRange(0.0001, 999999999); self.sa_param_stop.setValue(current_params['p_stop'])
        self.sa_param_stop.setToolTip("Valoarea numerică pentru criteriul de oprire setat mai sus.")
        layout.addRow(self.lbl_sa_stop, self.sa_param_stop)
        
        self.update_sa_ui()
        
        buttons = QDialogButtonBox()
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")
        buttons.addButton(btn_ok, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(btn_cancel, QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
    def update_sa_ui(self):
        cool = self.sa_cooling.currentText()
        if cool == 'geometric': self.lbl_sa_racire.setText("Alpha (0.9 - 0.99):")
        elif cool == 'liniara': self.lbl_sa_racire.setText("Scădere Delta:")
        elif cool == 'logaritmica': self.lbl_sa_racire.setText("Constanta c:")

        stop = self.sa_stop.currentText()
        if stop == 'iteratii': self.lbl_sa_stop.setText("Număr maxim de iterații:")
        elif stop == 't_min': self.lbl_sa_stop.setText("Temp de îngheț (T_min):")
        elif stop == 'stagnare': self.lbl_sa_stop.setText("Sistare la X stagnări:")
        
    def get_params(self):
        return {
            't_max': self.sa_tmax.value(),
            'mod_init': self.sa_init.currentText(),
            'cooling': self.sa_cooling.currentText(),
            'p_racire': self.sa_param_racire.value(),
            'terminare': self.sa_stop.currentText(),
            'p_stop': self.sa_param_stop.value()
        }


class GASettingsDialog(QDialog):
    def __init__(self, current_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setări Avansate Algoritm Genetic")
        self.setMinimumWidth(280)
        layout = QFormLayout(self)
        
        self.ga_pop = QSpinBox(); self.ga_pop.setRange(2, 999999999); self.ga_pop.setValue(current_params['pop_size'])
        self.ga_pop.setToolTip("Numărul de soluții candidate (cromozomi) prezente într-o singură generație. Mai multe = diversitate mai mare.")
        self.ga_gen = QSpinBox(); self.ga_gen.setRange(1, 999999999); self.ga_gen.setValue(current_params['n_generatii'])
        self.ga_gen.setToolTip("Numărul de generații pe care se va întinde evoluția (iterațiile algoritmului).")
        self.ga_mut = QSpinBox(); self.ga_mut.setRange(0, 100); self.ga_mut.setValue(current_params['rata_mutatie'])
        self.ga_mut.setToolTip("Probabilitatea (0-100%) ca o soluție rezultată să sufere o mutație minoră. Ajută la evitarea blocajelor locale.")
        
        self.ga_sel = QComboBox()
        self.ga_sel.addItems(['tournament', 'rws', 'rank', 'sss', 'random'])
        self.ga_sel.setCurrentText(current_params.get('selectie', 'tournament'))
        from PySide6.QtCore import Qt
        self.ga_sel.setItemData(0, "Tournament: Se aleg K indivizi și se luptă; cel mai bun devine părinte.", Qt.ItemDataRole.ToolTipRole)
        self.ga_sel.setItemData(1, "RWS: Șanse proporționale cu calitatea rutei (ca o roată de noroc).", Qt.ItemDataRole.ToolTipRole)
        self.ga_sel.setItemData(2, "Rank: Selecție bazată pe poziția în clasament (1, 2, 3...), ignorând diferențele imense de scor.", Qt.ItemDataRole.ToolTipRole)
        self.ga_sel.setItemData(3, "SSS (Steady-State): Indivizii de jos sunt înlocuiți imediat de puii celor de sus.", Qt.ItemDataRole.ToolTipRole)
        self.ga_sel.setItemData(4, "Random: Părinții sunt aleși 100% aleator (bun pentru comparații/teste).", Qt.ItemDataRole.ToolTipRole)
        self.ga_sel.setToolTip("Mecanismul prin care se aleg părinții. Extinde lista pentru detalii.")
        
        self.lbl_ga_k = QLabel("K Turneu:")
        self.ga_k = QSpinBox(); self.ga_k.setRange(1, 100); self.ga_k.setValue(current_params.get('k_tournament', 3))
        self.ga_k.setToolTip("Câți indivizi aleatori intră în ringul unui singur turneu de reproducere (folosit doar la selecția Tournament).")
        
        self.ga_elit = QSpinBox(); self.ga_elit.setRange(0, 999999999); self.ga_elit.setValue(current_params.get('elitism', 2))
        self.ga_elit.setToolTip("Câți din cei mai performanți indivizi absoluți sunt copiați nemodificați în următoarea generație pentru a nu se pierde aurul.")
        
        layout.addRow("Populație:", self.ga_pop)
        layout.addRow("Generații:", self.ga_gen)
        layout.addRow("Mutație (%):", self.ga_mut)
        layout.addRow("Selecție:", self.ga_sel)
        layout.addRow(self.lbl_ga_k, self.ga_k)
        layout.addRow("Elitism:", self.ga_elit)
        
        self.ga_sel.currentIndexChanged.connect(self.update_ga_ui)
        self.update_ga_ui()
        
        buttons = QDialogButtonBox()
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")
        buttons.addButton(btn_ok, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(btn_cancel, QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
    def update_ga_ui(self):
        if self.ga_sel.currentText() == "tournament":
            self.lbl_ga_k.show(); self.ga_k.show()
        else:
            self.lbl_ga_k.hide(); self.ga_k.hide()
            
    def get_params(self):
        return {
            'pop_size': self.ga_pop.value(),
            'n_generatii': self.ga_gen.value(),
            'rata_mutatie': self.ga_mut.value(),
            'selectie': self.ga_sel.currentText(),
            'k_tournament': self.ga_k.value(),
            'elitism': self.ga_elit.value()
        }

class HCSettingsDialog(QDialog):
    def __init__(self, current_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setări Avansate Hill Climbing")
        self.setMinimumWidth(280)
        layout = QFormLayout(self)
        
        self.hc_restarts = QSpinBox()
        self.hc_restarts.setRange(1, 999999999)
        self.hc_restarts.setValue(current_params.get('restarts', 10))
        self.hc_restarts.setToolTip("Numărul de reporniri (random restarts) pe care le face algoritmul pentru a evita optimii locali.")
        
        self.hc_init = QComboBox()
        self.hc_init.addItems(['random', 'nn'])
        self.hc_init.setCurrentText(current_params.get('init_type', 'random'))
        
        self.hc_op = QComboBox()
        self.hc_op.addItems(['2-opt', 'swap', 'insert'])
        self.hc_op.setCurrentText(current_params.get('operator', '2-opt'))
        
        layout.addRow("Restarturi:", self.hc_restarts)
        layout.addRow("Stare Inițială:", self.hc_init)
        layout.addRow("Operator Vecini:", self.hc_op)
        
        buttons = QDialogButtonBox()
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")
        buttons.addButton(btn_ok, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(btn_cancel, QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
    def get_params(self):
        return {
            'restarts': self.hc_restarts.value(),
            'init_type': self.hc_init.currentText(),
            'operator': self.hc_op.currentText()
        }