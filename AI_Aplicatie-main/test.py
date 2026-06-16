import sys
import random
import math
import os 
import json
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QTextEdit, QGraphicsScene, QGraphicsView,
    QSplitter, QFrame, QSpinBox, QComboBox, QStackedWidget, QDoubleSpinBox, 
    QMessageBox, QLineEdit, QFileDialog, QListWidget, QTableWidget, 
    QTableWidgetItem, QHeaderView, QInputDialog
)
from PySide6.QtGui import QPen, QBrush, QColor, QPainter, QPolygonF
from PySide6.QtCore import Qt, QPointF


from algo.nlp_classifier import ClasificatorText
from sklearn.datasets import fetch_20newsgroups
# --- Importuri Canvas Matplotlib pentru Embed în PySide6 ---
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from algo.nearest_neighbor import rezolva_tsp_nn
from algo.backtracking_tsp import BacktrackingTSP
from algo.simulated_annealing_tsp import rezolva_tsp_sa_custom
from algo.genetic_tsp import TSPGenetic

# Asigură-te că ai importurile necesare la începutul fișierului:
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

from NLPTab import NLPTab
from TSPTab import TSPTab
from BenchmarkTab import  BenchmarkTab





# =========================================================
# FEREASTRA PRINCIPALĂ CARE LEAGĂ TOATE TAB-URILE
# =========================================================
class AIEducationalApp(QMainWindow):
   def __init__(self):
        super().__init__()
        self.setWindowTitle("Hub Educațional - Analiza Convergenței & NLP")
        self.setMinimumSize(1250, 750)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.tsp_tab = TSPTab()
        self.tabs.addTab(self.tsp_tab, "Comis-Voiajor (TSP)")
        
        # ADAUGĂ TAB-UL DE BENCHMARK AICI
        self.bench_tab = BenchmarkTab()
        self.tabs.addTab(self.bench_tab, "Benchmarking Algoritmi")
        
        self.nlp_tab = NLPTab()
        self.tabs.addTab(self.nlp_tab, "Procesarea Limbajului Natural (NLP)")

# =========================================================
# QSS GLOBAL (STYLE SHEET)
# Tema inspirată din mediile moderne de dezvoltare (Catppuccin Macchiato)
# =========================================================
GLOBAL_QSS = """
/* Setări Generale */
QWidget {
    background-color: #24273A;
    color: #CAD3F5;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}

/* Containere și Frame-uri */
QFrame {
    background-color: #363A4F;
    border: 1px solid #494D64;
    border-radius: 8px;
    padding: 5px;
}

/* Etichete (Labels) */
QLabel {
    background-color: transparent;
    border: none;
    font-size: 14px;
}
QLabel b {
    color: #8AADF4;
}
QLabel h2 {
    color: #8AADF4;
    font-size: 18px;
}

/* Tab-uri (QTabWidget) */
QTabWidget::pane {
    border: 1px solid #494D64;
    border-radius: 5px;
    background: #24273A;
    top: -1px;
}
QTabBar::tab {
    background: #363A4F;
    color: #8087A2;
    padding: 8px 16px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border: 1px solid #494D64;
    border-bottom: none;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #24273A;
    color: #8AADF4;
    font-weight: bold;
    border-bottom: 2px solid #24273A;
}
QTabBar::tab:hover:!selected {
    background: #494D64;
    color: #CAD3F5;
}

/* Controale Input (SpinBox, ComboBox, LineEdit) */
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background-color: #1E2030;
    border: 1px solid #5B6078;
    border-radius: 4px;
    padding: 4px;
    color: #CAD3F5;
}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {
    border: 1px solid #8AADF4;
}

/* Butoane Standard */
QPushButton {
    background-color: #494D64;
    color: #CAD3F5;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #5B6078;
}
QPushButton:pressed {
    background-color: #363A4F;
}

/* Butoane Specifice: Run (Succes) */
QPushButton#btn_run {
    background-color: #A6DA95;
    color: #24273A;
    font-size: 14px;
    padding: 10px;
}
QPushButton#btn_run:hover {
    background-color: #8BD5CA;
}
QPushButton#btn_run:disabled {
    background-color: #494D64;
    color: #8087A2;
}

/* Butoane Specifice: Stop (Pericol) */
QPushButton#btn_stop {
    background-color: #ED8796;
    color: #24273A;
    font-size: 14px;
    padding: 10px;
}
QPushButton#btn_stop:hover {
    background-color: #F5BDE6;
}
QPushButton#btn_stop:disabled {
    background-color: #494D64;
    color: #8087A2;
}

/* Butoane Specifice: Acțiuni (Albastru) */
QPushButton#btn_action {
    background-color: #8AADF4;
    color: #24273A;
    padding: 8px;
}
QPushButton#btn_action:hover {
    background-color: #B7BDF8;
}

/* Jurnal / Console / Table */
QTextEdit, QTableWidget {
    background-color: #181825;
    color: #A6DA95;
    border: 1px solid #363A4F;
    border-radius: 4px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
}
QTableWidget {
    color: #CAD3F5;
    gridline-color: #363A4F;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QHeaderView::section {
    background-color: #363A4F;
    color: #CAD3F5;
    padding: 4px;
    border: 1px solid #494D64;
    font-weight: bold;
}

/* Progres */
QProgressBar {
    border: 1px solid #494D64;
    border-radius: 4px;
    background-color: #1E2030;
    text-align: center;
    color: #CAD3F5;
}
QProgressBar::chunk {
    background-color: #C6A0F6;
    border-radius: 3px;
}
"""

# =========================================================
# MOTORUL CARE PORNEȘTE APLICAȚIA
# =========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    app.setStyleSheet(GLOBAL_QSS) # Aplicăm tema QSS
    
    window = AIEducationalApp()
    window.show()
    sys.exit(app.exec())