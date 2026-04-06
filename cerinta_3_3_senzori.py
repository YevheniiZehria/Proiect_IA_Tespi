"""
Cerința 3.3 + Mișcare - Monitorizare senzori și deplasare Pioneer P3-DX.
IA Lab #06 - Inteligență Artificială 2025-2026
"""
import time
import os
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

SENSOR_MAX_RANGE = 1.0  # metri
V_FORWARD = 2.0         # rad/s

SENSOR_LABELS = [
    "S00  fata-stanga-ext ", "S01  fata-stanga     ", "S02  fata-centru-st  ",
    "S03  fata-centru-st  ", "S04  fata-centru-dr  ", "S05  fata-centru-dr  ",
    "S06  fata-dreapta     ", "S07  fata-dreapta-ext", "S08  lateral-dreapta ",
    "S09  lateral-dreapta ", "S10  spate-dreapta   ", "S11  spate-centru    ",
    "S12  spate-centru    ", "S13  spate-stanga    ", "S14  lateral-stanga  ",
    "S15  lateral-stanga  ",
]

def read_all_sensors(sim, sensors):
    readings = []
    for sensor in sensors:
        result, distance, *_ = sim.readProximitySensor(sensor)
        detected = bool(result)
        dist = distance if detected else SENSOR_MAX_RANGE
        readings.append((detected, dist))
    return readings

def print_dashboard(readings, v_left, v_right):
    """Afiseaza dashboard-ul si viteza actuala a motoarelor."""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=== DASHBOARD SENZORI & MISCARE - Pioneer P3-DX ===")
    print(f"Status Motoare: L={v_left:.2f} rad/s, R={v_right:.2f} rad/s\n")
    print(f"  {'Idx':<5} {'Eticheta':<25} {'Detectat':<10} {'Distanta':>10}  {'Bar':}")
    print("  " + "-" * 65)
    for i, (detected, dist) in enumerate(readings):
        bar_len = int((1.0 - dist / SENSOR_MAX_RANGE) * 20) if detected else 0
        bar = "█" * bar_len
        dist_str = f"{dist:.3f} m" if detected else "(nimic)"
        det_str = "DA" if detected else "nu"
        print(f"  [{i:2d}]  {SENSOR_LABELS[i]}  {det_str:<10} {dist_str:>10}  {bar}")
    print("\n  Ctrl+C pentru a opri robotul si simularea.")

def main():
    client = RemoteAPIClient()
    sim = client.require('sim')

    # Obtinere handle-uri
    robot = sim.getObject('/PioneerP3DX')
    left_motor = sim.getObject('/PioneerP3DX/leftMotor')
    right_motor = sim.getObject('/PioneerP3DX/rightMotor')
    sensors = [sim.getObject(f'/PioneerP3DX/ultrasonicSensor[{i}]') for i in range(16)]

    # Activam modul sincron pentru a nu avea decalaj intre senzori si miscare
    sim.setStepping(True)
    sim.startSimulation()

    print("Simulare pornita. Robotul se misca si monitorizeaza senzorii...")

    try:
        while True:
            # 1. PERCEPTIE: Citim senzorii
            readings = read_all_sensors(sim, sensors)
            
            # 2. ACTIUNE: Seteaza viteza (Mers inainte)
            # Poti schimba vitezele aici pentru a testa viraje
            v_l, v_r = V_FORWARD, V_FORWARD
            sim.setJointTargetVelocity(left_motor, v_l)
            sim.setJointTargetVelocity(right_motor, v_r)

            # 3. VIZUALIZARE: Afisam datele
            print_dashboard(readings, v_l, v_r)
            
            # 4. PAS SIMULARE: Avansam simularea
            # Facem 5 pasi de simulare (5 * 50ms = 0.25s de miscare per refresh)
            for _ in range(5):
                sim.step()

    except KeyboardInterrupt:
        print("\nIntrerupere primita. Oprire robot...")
    finally:
        # Curatenie la iesire: oprim motoarele si simularea
        sim.setJointTargetVelocity(left_motor, 0)
        sim.setJointTargetVelocity(right_motor, 0)
        sim.stopSimulation()
        print("Simulare finalizata.")

if __name__ == '__main__':
    main()