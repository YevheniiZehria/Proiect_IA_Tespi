"""
Cerința 3.4 - Comportament reactiv simplu: oprire la obstacol.
IA Lab #06 - Inteligență Artificială 2025-2026
"""
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

V_FORWARD     = 2.0    # rad/s - viteza de deplasare inainte
STOP_DISTANCE = 0.5    # metri - distanta la care robotul opreste
FRONT_SENSORS = [2, 3, 4, 5]  # indicii senzorilor frontali (ajustati dupa cerinta 3.3)
SENSOR_MAX    = 1.0    # metri - valoare returnata cand senzorul nu detecteaza nimic


def get_min_front_distance(sim, sensors, front_indices):
    """
    Returneaza distanta minima detectata de senzorii frontali.

    Args:
        sim: obiectul API CoppeliaSim.
        sensors: lista completa de handle-uri senzori.
        front_indices: lista indicilor senzorilor de monitorizat.

    Returns:
        float: distanta minima in metri (SENSOR_MAX daca nimic detectat).
    """
    min_dist = SENSOR_MAX
    for idx in front_indices:
        result, distance, *_ = sim.readProximitySensor(sensors[idx])
        if result and distance < min_dist:
            min_dist = distance
    return min_dist


def main():
    client = RemoteAPIClient()
    sim = client.require('sim')

    # 1. Obținere Handle-uri
    left_motor  = sim.getObject('/PioneerP3DX/leftMotor')
    right_motor = sim.getObject('/PioneerP3DX/rightMotor')
    sensors     = [
        sim.getObject(f'/PioneerP3DX/ultrasonicSensor[{i}]')
        for i in range(16)
    ]

    # 2. Configurare Mod Sincron pentru reacție rapidă
    sim.setStepping(True)
    sim.startSimulation()
    
    print(f"Robot pornit (Mod Sincron).")
    print(f"Se opreste la obstacol < {STOP_DISTANCE} m. (Ctrl+C pentru iesire)")

    try:
        while True:
            # A. PERCEPȚIE: Citim distanța minimă de la senzorii frontali [2, 3, 4, 5]
            dist_front = get_min_front_distance(sim, sensors, FRONT_SENSORS)

            # B. DECIZIE & ACȚIUNE
            if dist_front < STOP_DISTANCE:
                # Logica de OPRIRE
                sim.setJointTargetVelocity(left_motor, 0.0)
                sim.setJointTargetVelocity(right_motor, 0.0)
                # Folosim \r pentru a nu umple consola, scriem peste linia anterioară
                print(f"[STATUS] STOP | Obstacol detectat la: {dist_front:.3f} m    ", end='\r')
            else:
                # Logica de MERS ÎNAINTE
                sim.setJointTargetVelocity(left_motor, V_FORWARD)
                sim.setJointTargetVelocity(right_motor, V_FORWARD)
                print(f"[STATUS] MERS | Cale libera ({dist_front:.3f} m)         ", end='\r')

            # C. SINCRONIZARE: Avansăm simularea cu un pas (50ms)
            sim.step()

    except KeyboardInterrupt:
        print("\n\nOprire manuala (Ctrl+C).")
    except Exception as e:
        print(f"\nA aparut o eroare: {e}")
    finally:
        # Curățenie obligatorie
        sim.setJointTargetVelocity(left_motor, 0.0)
        sim.setJointTargetVelocity(right_motor, 0.0)
        sim.stopSimulation()
        print("Simulare finalizata.")


if __name__ == '__main__':
    main()