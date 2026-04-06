import time
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# --- Parametri Braitenberg ---
V_BASE = 3.0
V_MAX = 6.0
K_SENSOR = 6.0
SENSOR_MAX = 1.0

WEIGHTS = [
    (+0.5, -0.5), (+1.0, -1.0), (+1.5, -1.5), (+2.0, -2.0),
    (-2.0, +2.0), (-1.5, +1.5), (-1.0, +1.0), (-0.5, +0.5)
]

def get_sensor_readings(sim, sensors):
    """Returnează valorile normalizate ale senzorilor (0-1)."""
    readings = []
    for i in range(8):
        result, distance, *_ = sim.readProximitySensor(sensors[i])
        proximity = 1.0 - (distance / SENSOR_MAX) if result else 0.0
        readings.append(max(0.0, min(1.0, proximity)))
    return readings

def save_plots(data):
    """Generează cele 3 grafice cerute la finalul rulării."""
    if not data: return
    
    # Convertim lista de dicționare în array-uri numpy pentru manipulare ușoară
    t = [d['timestamp'] for d in data]
    vx = [d['pos_x'] for d in data]
    vy = [d['pos_y'] for d in data]
    vl = [d['v_left'] for d in data]
    vr = [d['v_right'] for d in data]
    
    # Matrice pentru heatmap (senzorii s0-s7 pe rânduri)
    sensors_matrix = np.array([[d[f's{i}'] for d in data] for i in range(8)])

    # 1. Traiectoria XY
    plt.figure(figsize=(8, 6))
    plt.plot(vx, vy, 'b-', label='Traiectorie')
    plt.scatter(vx[0], vy[0], color='green', label='Start')
    plt.scatter(vx[-1], vy[-1], color='red', label='Final')
    plt.title('Traiectoria Robotului în Planul XY')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.grid(True)
    plt.savefig('tema_b_traiectorie.png')
    plt.close()

    # 2. Viteze în funcție de timp
    plt.figure(figsize=(10, 5))
    plt.plot(t, vl, label='Viteza Stânga')
    plt.plot(t, vr, label='Viteza Dreapta')
    plt.title('Vitezele Motoarelor în Timp')
    plt.xlabel('Timp [s]')
    plt.ylabel('Viteza [rad/s]')
    plt.legend()
    plt.savefig('tema_b_viteze.png')
    plt.close()

    # 3. Heatmap Senzori
    plt.figure(figsize=(12, 4))
    plt.imshow(sensors_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(label='Intensitate Detectie (Proximitate)')
    plt.title('Heatmap Activare Senzori (S0-S7)')
    plt.xlabel('Iterație / Timp')
    plt.ylabel('Index Senzor')
    plt.savefig('tema_b_heatmap.png')
    plt.close()
    
    print("\nGraficele au fost salvate ca PNG.")

def main():
    client = RemoteAPIClient()
    sim = client.require('sim')

    # Obținere handle-uri
    robot = sim.getObject('/PioneerP3DX')
    left_motor = sim.getObject('/PioneerP3DX/leftMotor')
    right_motor = sim.getObject('/PioneerP3DX/rightMotor')
    sensors = [sim.getObject(f'/PioneerP3DX/ultrasonicSensor[{i}]') for i in range(16)]

    # Pregătire fișier CSV
    os.makedirs('tema06', exist_ok=True)
    csv_file = open('tema06/log_braitenberg.csv', 'w', newline='')
    fieldnames = ['timestamp', 'v_left', 'v_right'] + [f's{i}' for i in range(8)] + ['pos_x', 'pos_y']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    collected_data = []
    sim.startSimulation()
    print("Simulare pornită. Colectez date... (Ctrl+C pentru stop)")

    try:
        while True:
            # 1. Percepție
            s_vals = get_sensor_readings(sim, sensors)
            pos = sim.getObjectPosition(robot, sim.handle_world)
            t_curr = sim.getSimulationTime()

            # 2. Calcul Viteze Braitenberg
            v_left = V_BASE + sum(W[0] * s for W, s in zip(WEIGHTS, s_vals)) * K_SENSOR
            v_right = V_BASE + sum(W[1] * s for W, s in zip(WEIGHTS, s_vals)) * K_SENSOR
            
            v_left = max(-V_MAX, min(V_MAX, v_left))
            v_right = max(-V_MAX, min(V_MAX, v_right))

            # 3. Acțiune
            sim.setJointTargetVelocity(left_motor, v_left)
            sim.setJointTargetVelocity(right_motor, v_right)

            # 4. Logare date
            row = {
                'timestamp': t_curr, 'v_left': v_left, 'v_right': v_right,
                'pos_x': pos[0], 'pos_y': pos[1]
            }
            for i, val in enumerate(s_vals): row[f's{i}'] = val
            
            writer.writerow(row)
            collected_data.append(row)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nOprit de utilizator.")
    finally:
        csv_file.close()
        sim.stopSimulation()
        print("Generez grafice...")
        save_plots(collected_data)

if __name__ == '__main__':
    main()