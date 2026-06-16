
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



class NLPTab(QWidget):
    def __init__(self):
        super().__init__()
        self.train_data = None
        self.test_data = None
        
        # Instanțiem modelul direct în memorie
        self.model_curent = ClasificatorText()
        self.rezultate_algoritmi = {} 
        self.date_custom_incarcate = False # Flag pentru a ști dacă avem date CSV
        
        self.build_ui()

    def build_ui(self):
        layout = QHBoxLayout(self)

        # ==========================================
        # PANOU STÂNGA (QTabWidget)
        # ==========================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)

        self.tabs_stanga = QTabWidget()
        left_layout.addWidget(self.tabs_stanga)

        # ------------------------------------------
        # SUB-TAB 1: Antrenare și Rulare
        # ------------------------------------------
        tab_run = QWidget()
        run_layout = QVBoxLayout(tab_run)
        run_layout.setSpacing(10)

        run_layout.addWidget(QLabel("<h2>Configurare Model</h2>"))
        
        run_layout.addWidget(QLabel("<b>Alege Setul de Date pentru Test/Train:</b>"))
        self.combo_dataset = QComboBox()
        self.combo_dataset.addItems([
            "20 Newsgroups (4 categorii, Implicit)", 
            "AG News - Știri (4 categorii)",
            "Date Custom (Din CSV)"
        ])
        run_layout.addWidget(self.combo_dataset)

        run_layout.addWidget(QLabel("<b>Alege Algoritmul:</b>"))
        self.combo_model_nlp = QComboBox()
        self.combo_model_nlp.addItems([
            "Naive Bayes (MultinomialNB)", "Linear SVM", "Regresie Logistică", "Random Forest"
        ])
        run_layout.addWidget(self.combo_model_nlp)
        
        # Setări Algoritm
        params_frame = QFrame()
        params_layout = QVBoxLayout(params_frame)
        params_layout.addWidget(QLabel("<b>Hiperparametri TF-IDF:</b>"))
        self.combo_ngram = QComboBox()
        self.combo_ngram.addItems(["Unigrame (1, 1)", "Uni- și Bigrame (1, 2)", "Trigrame (1, 3)"])
        params_layout.addWidget(QLabel("N-gram range:"))
        params_layout.addWidget(self.combo_ngram)
        self.combo_features = QComboBox()
        self.combo_features.addItems(["Toate", "500", "1000", "5000"])
        params_layout.addWidget(QLabel("max_features:"))
        params_layout.addWidget(self.combo_features)
        run_layout.addWidget(params_frame)

        self.btn_classify = QPushButton("Analizează Textul")
        self.btn_classify.setObjectName("btn_run")
        self.btn_classify.clicked.connect(self.analyze_text)
        run_layout.addWidget(self.btn_classify)

        self.btn_compare = QPushButton("Compară Modelele Testate (Bar Chart)")
        self.btn_compare.clicked.connect(self.show_comparison)
        run_layout.addWidget(self.btn_compare)

        # Rezultat
        self.lbl_result = QLabel("Rezultat: (Așteptare text...)")
        self.lbl_result.setObjectName("lbl_result")
        self.lbl_result.setWordWrap(True)
        run_layout.addWidget(self.lbl_result)

        run_layout.addWidget(QLabel("<b>Testare Rapidă (după antrenare):</b>"))
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Introdu text aici și apasă 'Prezice'")
        self.text_input.setMinimumHeight(50)
        self.text_input.setMaximumHeight(80)
        run_layout.addWidget(self.text_input)
        
        self.btn_predict = QPushButton("Prezice Text")
        self.btn_predict.clicked.connect(self.predict_custom_text)
        run_layout.addWidget(self.btn_predict)
        self.lbl_custom_pred = QLabel("Rezultat: -")
        run_layout.addWidget(self.lbl_custom_pred)
        run_layout.addStretch()

        self.tabs_stanga.addTab(tab_run, "1. Antrenare & Rulare")

        # ------------------------------------------
        # SUB-TAB 2: Gestionare Date Custom
        # ------------------------------------------
        tab_data = QWidget()
        data_layout = QVBoxLayout(tab_data)
        data_layout.setSpacing(10)

        data_layout.addWidget(QLabel("<b>Creează sau Încarcă Setul tău de Date:</b>"))
        
        # Butoane CSV Layout
        csv_buttons_layout = QHBoxLayout()
        self.btn_import_csv = QPushButton("Importă CSV")
        self.btn_import_csv.setObjectName("btn_action")
        self.btn_import_csv.clicked.connect(self.incarca_csv)
        
        self.btn_export_csv = QPushButton("Salvează CSV")
        self.btn_export_csv.setObjectName("btn_action")
        self.btn_export_csv.clicked.connect(self.salveaza_csv)
        
        csv_buttons_layout.addWidget(self.btn_import_csv)
        csv_buttons_layout.addWidget(self.btn_export_csv)
        data_layout.addLayout(csv_buttons_layout)

        self.lbl_csv_status = QLabel("Stare: Niciun fișier importat.")
        self.lbl_csv_status.setObjectName("lbl_status")
        data_layout.addWidget(self.lbl_csv_status)

        # Tabel pentru vizualizare si editare
        self.tabel_date = QTableWidget(0, 2)
        self.tabel_date.setHorizontalHeaderLabels(["Text Articol", "Categorie"])
        self.tabel_date.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tabel_date.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        data_layout.addWidget(self.tabel_date)

        # Adăugare manuală
        row_add = QHBoxLayout()
        self.input_text_nou = QLineEdit()
        self.input_text_nou.setPlaceholderText("Introdu textul...")
        self.input_cat_noua = QLineEdit()
        self.input_cat_noua.setPlaceholderText("Categorie (ex. sport)")
        
        self.btn_add_row = QPushButton("Adaugă Rând")
        self.btn_add_row.clicked.connect(self.adauga_rand_tabel)
        
        self.btn_del_row = QPushButton("Șterge Selectate")
        self.btn_del_row.clicked.connect(self.sterge_rand_tabel)
        
        row_add.addWidget(self.input_text_nou)
        row_add.addWidget(self.input_cat_noua)
        row_add.addWidget(self.btn_add_row)
        row_add.addWidget(self.btn_del_row)
        data_layout.addLayout(row_add)
        
        self.btn_aplica_date = QPushButton("Aplică Datele în Clasificator")
        self.btn_aplica_date.setObjectName("btn_run")
        self.btn_aplica_date.clicked.connect(self.aplica_date_tabel)
        data_layout.addWidget(self.btn_aplica_date)

        self.tabs_stanga.addTab(tab_data, "2. Gestionare Date")

        # ==========================================
        # PANOU DREAPTA (Grafice Matplotlib)
        # ==========================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Grafice de Evaluare (Așteptare Date...)")
        right_layout.addWidget(self.canvas)

        self.btn_save_plot = QPushButton("Salvează Grafic")
        self.btn_save_plot.clicked.connect(self.salveaza_grafic)
        right_layout.addWidget(self.btn_save_plot)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)
        self.splitter.setSizes([450, 550])
        layout.addWidget(self.splitter)

    # ==========================================
    # LOGICĂ DATE CUSTOM (TABEL, CSV)
    # ==========================================
    def incarca_csv(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Selectează Dataset CSV", "", "CSV Files (*.csv)")
        
        if filepath:
            try:
                documente_adaugate, categorii_gasite = self.model_curent.incarca_date_din_csv(filepath)
                
                self.tabel_date.setRowCount(0)
                for txt, cat in documente_adaugate:
                    row_position = self.tabel_date.rowCount()
                    self.tabel_date.insertRow(row_position)
                    self.tabel_date.setItem(row_position, 0, QTableWidgetItem(txt))
                    self.tabel_date.setItem(row_position, 1, QTableWidgetItem(cat))
                
                self.date_custom_incarcate = True
                nume_fisier = filepath.split('/')[-1]
                self.lbl_csv_status.setText(f"Încărcat: {nume_fisier} ({len(documente_adaugate)} rânduri)")
                self.combo_dataset.setCurrentIndex(1)
                
            except Exception as e:
                print(e.args)
                self.date_custom_incarcate = False
                self.lbl_csv_status.setText("Eroare la încărcare.")

    def salveaza_csv(self):
        if self.tabel_date.rowCount() == 0:
            QMessageBox.warning(self, "Tabel Gol", "Nu există date de salvat!")
            return
            
        filepath, _ = QFileDialog.getSaveFileName(self, "Salvează Dataset CSV", "", "CSV Files (*.csv)")
        if filepath:
            import csv
            with open(filepath, mode='w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['text', 'clasificare']) # Header așteptat de model
                for row in range(self.tabel_date.rowCount()):
                    txt = self.tabel_date.item(row, 0).text()
                    cat = self.tabel_date.item(row, 1).text()
                    writer.writerow([txt, cat])
            self.lbl_csv_status.setText(f"Salvat cu succes: {filepath.split('/')[-1]}")

    def adauga_rand_tabel(self):
        txt = self.input_text_nou.text().strip()
        cat = self.input_cat_noua.text().strip()
        
        if not txt or not cat:
            QMessageBox.warning(self, "Eroare", "Te rog introdu și textul și categoria.")
            return
            
        row_position = self.tabel_date.rowCount()
        self.tabel_date.insertRow(row_position)
        self.tabel_date.setItem(row_position, 0, QTableWidgetItem(txt))
        self.tabel_date.setItem(row_position, 1, QTableWidgetItem(cat))
        
        self.input_text_nou.clear()
        self.input_cat_noua.clear()

    def sterge_rand_tabel(self):
        selected_ranges = self.tabel_date.selectedRanges()
        if not selected_ranges:
            return
        # Stergem in ordine inversa
        rows_to_delete = set()
        for r in selected_ranges:
            for i in range(r.topRow(), r.bottomRow() + 1):
                rows_to_delete.add(i)
                
        for row in sorted(list(rows_to_delete), reverse=True):
            self.tabel_date.removeRow(row)

    def aplica_date_tabel(self):
        if self.tabel_date.rowCount() < 4:
            QMessageBox.warning(self, "Prea puține date", "Ai nevoie de cel puțin 4 rânduri pentru antrenare.")
            return
            
        self.model_curent.custom_texts = []
        self.model_curent.custom_categories = []
        
        for row in range(self.tabel_date.rowCount()):
            txt = self.tabel_date.item(row, 0).text()
            cat = self.tabel_date.item(row, 1).text()
            self.model_curent.custom_texts.append(txt)
            self.model_curent.custom_categories.append(cat)
            
        self.date_custom_incarcate = True
        self.lbl_csv_status.setText(f"Date aplicate manual ({self.tabel_date.rowCount()} rânduri)")
        self.combo_dataset.setCurrentIndex(1)
        QMessageBox.information(self, "Succes", "Datele au fost transferate cu succes către model!")

    # ==========================================
    # LOGICĂ RULARE / ANALIZĂ
    # ==========================================
    def analyze_text(self):
        algo_mapping = {0: "naive_bayes", 1: "svm", 2: "logreg", 3: "random_forest"}
        ngram_mapping = {0: (1, 1), 1: (1, 2), 2: (1, 3)}
        features_mapping = {0: None, 1: 500, 2: 1000, 3: 5000}
        
        nume_algoritm_vizual = self.combo_model_nlp.currentText()
        set_ales = self.combo_dataset.currentText()

        algo_ales = algo_mapping[self.combo_model_nlp.currentIndex()]
        ngram_ales = ngram_mapping[self.combo_ngram.currentIndex()]
        feat_ales = features_mapping[self.combo_features.currentIndex()]

        self.btn_classify.setEnabled(False)
        self.lbl_result.setText("<b>Status:</b> Se încarcă datele și se evaluează...")
        QApplication.processEvents()

        try:
            # Verificăm dacă dataset-ul a fost schimbat pentru a forța reîncărcarea
            if hasattr(self, 'last_dataset') and getattr(self, 'last_dataset') != set_ales:
                self.train_data = None
                self.test_data = None
            self.last_dataset = set_ales

            if "20 Newsgroups" in set_ales:
                # Logica standard (Dataset de test online)
                if self.train_data is None:
                    categorii_lab = ['sci.space', 'rec.sport.hockey', 'talk.politics.guns', 'comp.graphics']
                    self.train_data = fetch_20newsgroups(subset='train', categories=categorii_lab, remove=('headers', 'footers', 'quotes'))
                    self.test_data = fetch_20newsgroups(subset='test', categories=categorii_lab, remove=('headers', 'footers', 'quotes'))
                
                # Inițializăm un model nou pentru 20 newsgroups
                model_temp = ClasificatorText(algoritm=algo_ales, ngram_range=ngram_ales, max_features=feat_ales)
                train_x, train_y = self.train_data.data, self.train_data.target
                test_x, test_y = self.test_data.data, self.test_data.target
                target_names = self.train_data.target_names
                
                # Antrenăm modelul temporar și îl salvăm ca model curent
                acc, pred, cm = model_temp.antreneaza_si_evalueaza(train_x, train_y, test_x, test_y, target_names)
                self.model_curent = model_temp
                
            elif set_ales == "AG News - Știri (4 categorii)":
                if not getattr(self, 'agnews_dataset_loaded', False):
                    self.lbl_result.setText("<b>Status:</b> Se descarcă AG News...")
                    QApplication.processEvents()
                    import urllib.request
                    import csv
                    url = 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv'
                    response = urllib.request.urlopen(url)
                    lines = response.read().decode('utf-8').splitlines()
                    self.ag_texts = []
                    self.ag_categories = []
                    mapping = {"1": "World", "2": "Sports", "3": "Business", "4": "Sci/Tech"}
                    
                    reader = csv.reader(lines)
                    for row in reader:
                        if len(row) >= 3:
                            cat = mapping.get(row[0], "Unknown")
                            text = row[1] + " " + row[2]
                            self.ag_categories.append(cat)
                            self.ag_texts.append(text)
                    self.agnews_dataset_loaded = True
                
                model_temp = ClasificatorText(algoritm=algo_ales, ngram_range=ngram_ales, max_features=feat_ales)
                model_temp.custom_texts = self.ag_texts
                model_temp.custom_categories = self.ag_categories
                
                date_train, date_test = model_temp.pregateste_date_custom(test_size=0.2)
                train_x, train_y = date_train.data, date_train.target
                test_x, test_y = date_test.data, date_test.target
                target_names = date_train.target_names
                acc, pred, cm = model_temp.antreneaza_si_evalueaza(train_x, train_y, test_x, test_y, target_names)
                self.model_curent = model_temp

            else:
                # Logica Date Custom (CSV Memorie)
                if not self.date_custom_incarcate:
                    raise ValueError("Te rog să încarci un fișier CSV din tab-ul 'Date CSV' mai întâi.")
                
                
                # Actualizăm parametrii modelului curent care deține deja datele în liste
                self.model_curent.reconstruieste_pipeline(algoritm=algo_ales, ngram_range=ngram_ales, max_features=feat_ales)
                # Reatașăm datele deoarece __init__ le resetează
                # (Dacă backend-ul tău încarcă json în init, asigură-te că ai scos apelul incarca_dataset_local() din nlp_classifier.py)
                
                date_train, date_test = self.model_curent.pregateste_date_custom(test_size=0.2)
                
                train_x, train_y = date_train.data, date_train.target
                test_x, test_y = date_test.data, date_test.target
                target_names = date_train.target_names

                acc, pred, cm = self.model_curent.antreneaza_si_evalueaza(train_x, train_y, test_x, test_y, target_names)

            # Afișare Rezultate
            self.rezultate_algoritmi[nume_algoritm_vizual] = acc
            self.lbl_result.setText(f"<b>Acuratețe ({nume_algoritm_vizual}):</b> <span style='color:#2e7d32;'>{acc*100:.2f}%</span>")
            self.draw_confusion_matrix(cm, target_names, f"{nume_algoritm_vizual} - {set_ales}")

        except Exception as e:
            self.lbl_result.setText("<b>Eroare.</b> Verifică fișierul.")
            QMessageBox.warning(self, "Eroare la Antrenare", str(e))
        finally:
            self.btn_classify.setEnabled(True)

    def predict_custom_text(self):
        if not self.model_curent or not getattr(self.model_curent, 'is_trained', False):
            self.lbl_custom_pred.setText("Eroare: Antrenează modelul mai întâi!")
            return
        text = self.text_input.toPlainText().strip()
        if text:
            pred = self.model_curent.prezice(text)
            self.lbl_custom_pred.setText(f"<b>Rezultat:</b> <span style='color:#2e7d32;'>{pred}</span>")

    def draw_confusion_matrix(self, cm, display_labels, titlu):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=self.ax, cmap='Blues', colorbar=True)
        self.ax.set_title(f"Matrice Confuzie\n{titlu}")
        for tick in self.ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_horizontalalignment("right")
        self.fig.tight_layout()
        self.canvas.draw()

    def show_comparison(self):
        if not self.rezultate_algoritmi:
            self.lbl_result.setText("<b>Eroare:</b> Evaluează cel puțin un model.")
            return
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        modele = list(self.rezultate_algoritmi.keys())
        acurateti = list(self.rezultate_algoritmi.values())
        culori = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(modele)]
        bare = self.ax.bar(modele, acurateti, color=culori, edgecolor='black')
        self.ax.set_ylim(0, 1.05)
        self.ax.set_ylabel('Acuratețe')
        self.ax.set_title('Compararea Clasificatorilor Testati')
        for bar, val in zip(bare, acurateti):
            self.ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        for tick in self.ax.get_xticklabels():
            tick.set_rotation(15)
        self.fig.tight_layout()
        self.canvas.draw()

    def salveaza_grafic(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Salvează Grafic NLP", "", "Imagini PNG (*.png);;Imagini JPEG (*.jpg)")
        if filepath:
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, "Succes", f"Graficul a fost salvat cu succes în:\n{filepath}")