import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

class CustomDataBunch:
    def __init__(self, data, target, target_names):
        self.data = data
        self.target = target
        self.target_names = target_names

class ClasificatorText:
    def __init__(self, algoritm='svm', ngram_range=(1, 1), max_features=None):
        self.is_trained = False
        self.categorii = []
        self.algoritm_name = algoritm
        
        # Datele stau exclusiv în memoria RAM, populate de interfața grafică din CSV
        self.custom_texts = []
        self.custom_categories = []
        
        self.reconstruieste_pipeline(algoritm, ngram_range, max_features)

    def reconstruieste_pipeline(self, algoritm, ngram_range, max_features):
        """Permite schimbarea algoritmului (SVM, Naive Bayes etc.) fără a șterge textele din liste."""
        self.algoritm_name = algoritm
        if algoritm == 'naive_bayes':
            clf = MultinomialNB()
        elif algoritm == 'logreg':
            clf = LogisticRegression(max_iter=1000, solver='saga')
        elif algoritm == 'random_forest':
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        else:
            clf = LinearSVC(max_iter=2000)

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                stop_words='english',
                sublinear_tf=True
            )),
            ('clf', clf)
        ])
        self.is_trained = False

    def pregateste_date_custom(self, test_size=0.2):
        if len(self.custom_texts) < 4:
            raise ValueError("Adaugă cel puțin 4-5 documente pentru a putea antrena modelul.")
            
        target_names = list(set(self.custom_categories))
        targets_numeric = [target_names.index(cat) for cat in self.custom_categories]
        
        try:
            # Împărțim setul (Train/Test) - asigurându-ne că avem proporții corecte (stratify)
            X_train, X_test, y_train, y_test = train_test_split(
                self.custom_texts, targets_numeric, 
                test_size=test_size, random_state=42, stratify=targets_numeric
            )
        except ValueError:
             raise ValueError("Ai nevoie de cel puțin 2 rânduri CSV în FIECARE categorie pentru a putea testa.")

        train_data = CustomDataBunch(X_train, y_train, target_names)
        test_data = CustomDataBunch(X_test, y_test, target_names)
        return train_data, test_data

    def antreneaza_si_evalueaza(self, train_data, train_labels, test_data, test_labels, nume_categorii):
        self.categorii = nume_categorii
        self.pipeline.fit(train_data, train_labels)
        self.is_trained = True
        
        pred = self.pipeline.predict(test_data)
        acc = accuracy_score(test_labels, pred)
        cm = confusion_matrix(test_labels, pred)
        return acc, pred, cm

    def prezice(self, text):
        if not self.is_trained:
            return "Eroare: Modelul nu a fost antrenat încă!"
        pred_idx = self.pipeline.predict([text])[0]
        return self.categorii[pred_idx]
    

    def incarca_date_din_csv(self, cale_csv):
        import csv
        import os
        
        if not os.path.exists(cale_csv):
            raise FileNotFoundError(f"Nu am găsit fișierul CSV: {cale_csv}")

        # Resetăm datele vechi din memorie
        self.custom_texts = []
        self.custom_categories = []
        
        date_procesate = []
        categorii_unice = set()

        with open(cale_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            headers = reader.fieldnames
            if not headers or 'clasificare' not in headers or 'text' not in headers:
                raise ValueError("Fișierul CSV trebuie să conțină exact coloanele 'clasificare' și 'text'.")

            for row in reader:
                text_articol = row['text'].strip()
                categorie = row['clasificare'].strip()
                
                if text_articol and categorie:
                    # Le salvăm direct în listele modelului
                    self.custom_texts.append(text_articol)
                    self.custom_categories.append(categorie)
                    
                    # Le pregătim pentru a fi returnate interfeței grafice
                    date_procesate.append((text_articol, categorie))
                    categorii_unice.add(categorie)
                    
        return date_procesate, list(categorii_unice)