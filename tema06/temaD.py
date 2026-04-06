"""
Tema D (Bonus) - Vehicul Braitenberg: "Iubire" (Vehicul 3a).
Conexiuni ipsilaterale INHIBITORII.
"""
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

V_MAX      = 4.0   # Viteza de croazieră (când nu vede nimic)
SENSOR_MAX = 1.0   # Raza senzorilor

# Ponderi Inhibitorii (Ipsilaterale):
# Senzor stânga -> scade viteza motor stâng
# Senzor dreapta -> scade viteza motor drept
# Valorile sunt negative pentru a produce INHIBIȚIE.
WEIGHTS = [
    (-0.5,  0.0), # S0 fata-stanga-ext
    (-1.0,  0.0), # S1 fata-stanga
    (-2.0,  0.0), # S2 fata-centru-st
    (-3.0,  0.0), # S3 fata-centru-st
    ( 0.0, -3.0), # S4 fata-centru-dr
    ( 0.0, -2.0), # S5 fata-centru-dr
    ( 0.0, -1.0), # S6 fata-dreapta
    ( 0.0, -0.5), # S7 fata-dreapta-ext
]

def main():
    client = RemoteAPIClient()
    sim = client.require('sim')

    left_motor  = sim.getObject('/PioneerP3DX/leftMotor')
    right_motor = sim.getObject('/PioneerP3DX/rightMotor')
    sensors     = [sim.getObject(f'/PioneerP3DX/ultrasonicSensor[{i}]') for i in range(16)]

    sim.setStepping(True)
    sim.startSimulation()
    print("Vehicul 'Iubire' pornit. Robotul va cauta si va sta langa obstacole.")

    try:
        while True:
            v_left = V_MAX
            v_right = V_MAX

            for i, (w_l, w_r) in enumerate(WEIGHTS):
                res, dist, *_ = sim.readProximitySensor(sensors[i])
                if res:
                    # Normalizăm proximitatea (1 = foarte aproape, 0 = departe)
                    proximity = 1.0 - (dist / SENSOR_MAX)
                    
                    # Aplicăm inhibiția: viteza scade proporțional cu proximitatea
                    v_left  += w_l * proximity
                    v_right += w_r * proximity

            # Ne asigurăm că viteza nu devine negativă (robotul doar încetinește)
            v_left  = max(0.0, v_left)
            v_right = max(0.0, v_right)

            sim.setJointTargetVelocity(left_motor, v_left)
            sim.setJointTargetVelocity(right_motor, v_right)
            
            print(f"vL: {v_left:.2f} | vR: {v_right:.2f} (Status: Explorare/Atractie)", end='\r')
            sim.step()

    except KeyboardInterrupt:
        pass
    finally:
        sim.stopSimulation()

if __name__ == '__main__':
    main()