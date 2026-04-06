# Documentație Proiect: Navigație Reactivă Hibridă (Pioneer P3-DX)

Prezentul document detaliază implementarea modulului de control autonom pentru robotul Pioneer P3-DX, realizată în fișierul `copelia.py`. Arhitectura propusă se bazează pe un sistem de control reactiv care combină urmărirea conturului unui perete (wall-following) cu algoritmi de evitare a obstacolelor tip Braitenberg, oferind totodată o funcționalitate de cartografiere a traiectoriei.

---

## 1. Sistemul de Percepție

Robotul utilizează 16 senzori de proximitate ultrasonici pentru a mapa mediul înconjurător, având o rază maximă de detecție setată la 1.0 metri (`SENSOR_MAX_RANGE`).  Pentru a optimiza procesul decizional, senzorii sunt grupați logic în patru regiuni de interes:

* **Grup Frontal (`SENSORS_CENTER`)**: Indicii 2, 3, 4, 5.
* **Grup Stânga (`SENSORS_LEFT`)**: Indicii 0, 1, 14, 15.
* **Grup Dreapta (`SENSORS_RIGHT`)**: Indicii 6, 7, 8, 9.
* **Grup Spate (`SENSORS_BACK`)**: Indicii 10, 11, 12, 13.

La fiecare iterație a buclei de control (care rulează la o frecvență de 20 Hz prin funcția `time.sleep(0.05)`), scriptul interoghează toți senzorii și extrage distanța minimă înregistrată pentru fiecare grup, evaluând astfel starea de proximitate a obstacolelor pe cele patru axe principale.

---

## 2. Arhitectura de Control (Mașina de Stări)

Sistemul decizional este structurat ca o mașină de stări bazată pe priorități, evaluând condițiile de mediu de la cea mai critică la cea mai permisivă, în cadrul buclei `while True`:

### 2.1. Starea de Evadare (Prioritate Maximă)
Această stare este declanșată atunci când robotul este complet blocat de obstacole (distanța frontală < 0.3m, distanța la stânga < 0.4m, distanța la dreapta < 0.4m). Algoritmul identifică zona cu cel mai mult spațiu liber disponibil și aplică viteze opuse pe cele două motoare (ex. `v_left = -BASE_SPEED`, `v_right = BASE_SPEED`) pentru a executa o rotație rapidă pe loc și a părăsi blocajul.

### 2.2. Evitare Coliziune Frontală (Colț Interior)
Este activată proactiv dacă un obstacol frontal este detectat la mai puțin de 0.4m (variabila `FRONT_STOP`). Pentru a evita o coliziune iminentă, robotul reduce drastic viteza motorului stâng (`-BASE_SPEED * 0.5`) în timp ce o menține pe cea a motorului drept, forțând un viraj ascuțit de evitare către stânga.

### 2.3. Urmărire Perete (Wall-Following)
În absența unui pericol frontal critic, dacă robotul detectează un perete în partea sa dreaptă la o distanță mai mică de 0.8m, se activează un controller de tip Proporțional (P). Acesta calculează eroarea dintre distanța curentă măsurată și distanța țintă dorită (`TARGET_DIST = 0.5m`). Eroarea, înmulțită cu o constantă de proporționalitate (`K_P = 3.0`), este folosită pentru a ajusta fin vitezele roților, corectând deviațiile pentru a menține o traiectorie paralelă cu suprafața peretelui.

### 2.4. Navigație Liberă (Braitenberg)
Dacă nicio condiție restrictivă de mai sus nu este îndeplinită, robotul rulează logica de bază a vehiculelor Braitenberg.  Prin funcțiile `BraitSum` și `MotorLogic`, viteza fiecărei roți este ajustată dinamic pe baza unei sume ponderate a lecturilor senzorilor (folosind listele `leftBrait` și `rightBrait`), facilitând o evitare fluidă și reactivă a obstacolelor aflate la distanțe mai mari. 

La finalul fiecărei decizii, vitezele motoarelor sunt limitate superior și inferior (`cap = BASE_SPEED * 1.5`) pentru a asigura stabilitatea mecanică a simulării.

---

## 3. Colectarea Datelor și Generarea Traiectoriei

Un element adițional al scriptului este modulul de tracking spațial. Pe parcursul simulării, funcția `sim.getObjectPosition` extrage constant coordonatele absolute (X, Y) ale centrului de masă al robotului (`/PioneerP3DX`) relativ la coordonatele globale ale mediului (`sim.handle_world`). 

La întreruperea manuală a execuției scriptului (prin semnalul `KeyboardInterrupt` / Ctrl+C), codul intră în blocul `finally` unde oprește ambele motoare și încheie simularea. Imediat după, utilizează biblioteca `matplotlib.pyplot` pentru a procesa listele `path_x` și `path_y`, generând un grafic 2D complet al traseului. Graficul indică marcaje specifice (cerc verde pentru Start, cruce roșie pentru Stop) și folosește proporții egale pe axe (`plt.axis('equal')`) pentru o reprezentare geometrică precisă.

---

## 4. Instrucțiuni de Rulare și Configurare

**Dependențe Necesare:**
Mediul Python trebuie să includă pachetul oficial CoppeliaSim ZMQ și biblioteca Matplotlib:
```bash
pip install coppeliasim-zmqremoteapi-client matplotlib