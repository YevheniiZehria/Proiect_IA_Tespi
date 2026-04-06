
import time
import random
from enum import Enum, auto
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


V_FORWARD     = 2.0
V_BACKWARD    = -1.5
V_TURN        = 1.5
STOP_DISTANCE = 0.5   # Distanța critică pentru manevra de spate
DETECTION_DIST = 0.8  # Distanța de la care începe să ocolească ușor
FRONT_SENSORS = [2, 3, 4, 5]
SENSOR_MAX    = 1.0

class RobotState(Enum):
    FORWARD  = auto()
    BACKWARD = auto()
    TURNING  = auto()

def get_min_front_distance(sim, sensors, front_indices):
    min_dist = SENSOR_MAX
    for idx in front_indices:
        result, distance, *_ = sim.readProximitySensor(sensors[idx])
        if result and distance < min_dist:
            min_dist = distance
    return min_dist

def main():
    client = RemoteAPIClient()
    sim = client.require('sim')


    try:
        left_motor  = sim.getObject('/PioneerP3DX/leftMotor')
        right_motor = sim.getObject('/PioneerP3DX/rightMotor')
        sensors     = [sim.getObject(f'/PioneerP3DX/ultrasonicSensor[{i}]') for i in range(16)]
    except:
        print("Eroare: Asigurați-vă că scena este încărcată în CoppeliaSim!")
        return

    sim.setStepping(True)
    sim.startSimulation()
    
   
    current_state = RobotState.FORWARD
    state_start_time = 0
    turn_direction = 1 

    print(f"Robot pornit. Mod: Explorare Inteligentă. Stare: {current_state.name}")

    try:
        while True:
            dist_front = get_min_front_distance(sim, sensors, FRONT_SENSORS)
            sim_time = sim.getSimulationTime()

            
            
            if current_state == RobotState.FORWARD:
                if dist_front < STOP_DISTANCE:
                    
                    current_state = RobotState.BACKWARD
                    state_start_time = sim_time
                    print(f"\n[EVENT] Blocaj frontal ({dist_front:.2f}m) -> Recuperare (BACKWARD)")
                else:
                    
                    res_L, dist_L, *_ = sim.readProximitySensor(sensors[1])
                    res_R, dist_R, *_ = sim.readProximitySensor(sensors[6])
                    
                    v_l, v_r = V_FORWARD, V_FORWARD
                    
                   
                    if res_L and dist_L < DETECTION_DIST:
                        v_l = V_FORWARD + 0.5 # Accelerează roata stângă
                        v_r = V_FORWARD - 0.5 # Încetinește roata dreaptă
                    
                    
                    elif res_R and dist_R < DETECTION_DIST:
                        v_r = V_FORWARD + 0.5 # Accelerează roata dreaptă
                        v_l = V_FORWARD - 0.5 # Încetinește roata stângă

                    sim.setJointTargetVelocity(left_motor, v_l)
                    sim.setJointTargetVelocity(right_motor, v_r)

            elif current_state == RobotState.BACKWARD:

                if sim_time - state_start_time < 1.2:
                    sim.setJointTargetVelocity(left_motor, V_BACKWARD)
                    sim.setJointTargetVelocity(right_motor, V_BACKWARD)
                else:
                    current_state = RobotState.TURNING
                    state_start_time = sim_time
                    turn_direction = random.choice([1, -1]) # Stânga sau Dreapta
                    print(f"[EVENT] Spatiu creat -> Rotire ({'Stanga' if turn_direction==1 else 'Dreapta'})")

            elif current_state == RobotState.TURNING:
                
                if sim_time - state_start_time < 1.5:
                    sim.setJointTargetVelocity(left_motor, -V_TURN * turn_direction)
                    sim.setJointTargetVelocity(right_motor, V_TURN * turn_direction)
                else:
                    current_state = RobotState.FORWARD
                    print(f"[EVENT] Cale liberă -> Reluare FORWARD")

            
            print(f"[STATE]: {current_state.name:<10} | Dist: {dist_front:.3f}m", end='\r')

            sim.step()

    except KeyboardInterrupt:
        print("\n\nOprire manuală (Ctrl+C).")
    finally:
        
        sim.setJointTargetVelocity(left_motor, 0.0)
        sim.setJointTargetVelocity(right_motor, 0.0)
        sim.stopSimulation()
        print("Simulare finalizată.")

if __name__ == '__main__':
    main()