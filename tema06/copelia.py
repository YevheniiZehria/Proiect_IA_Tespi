import time
import os
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


leftBrait  = [0.5, 1.0, 1.5, 2.0, -3.0, -1.5, -1.0, -0.5]
rightBrait = [-0.5, -1.0, -1.5, -2.0, 2.0, 1.5, 1.0, 0.5]
K_CONSTANT = 0.5 


BASE_SPEED  = 2.0      # rad/s - viteza de baza
TARGET_DIST = 0.5      # m - distanta dorita fata de peretele drept
K_P         = 3.0      # coeficient proportional (P-controller)
FRONT_STOP  = 0.4      # m - distanta de declansare viraj la obstacol frontal
SENSOR_MAX_RANGE = 1.0 # m - valoare implicita cand senzorul nu detecteaza


SENSORS_LEFT   = [0, 1, 14, 15]
SENSORS_CENTER = [2, 3, 4, 5]
SENSORS_RIGHT  = [6, 7, 8, 9]
SENSORS_BACK   = [10, 11, 12, 13]

SENSOR_LABELS = [
    "S00  fata-stanga-ext ", "S01  fata-stanga     ", "S02  fata-centru-st  ", "S03  fata-centru-st  ",
    "S04  fata-centru-dr  ", "S05  fata-centru-dr  ", "S06  fata-dreapta    ", "S07  fata-dreapta-ext",
    "S08  lateral-dreapta ", "S09  lateral-dreapta ", "S10  spate-dreapta   ", "S11  spate-centru    ",
    "S12  spate-centru    ", "S13  spate-stanga    ", "S14  lateral-stanga  ", "S15  lateral-stanga  ",
]

def read_all_sensors(sim, sensors):
    
    readings = []
    for sensor in sensors:
        result, distance, *_ = sim.readProximitySensor(sensor)
        detected = bool(result)
        dist = distance if detected else SENSOR_MAX_RANGE
        readings.append((detected, dist))
    return readings

def get_group_min_distance(readings, indices):
  
    return min([readings[i][1] for i in indices])

def BraitSum(reading, vector):
   
    total_sum = 0
    for i in range(len(reading)):
        val, distance = reading[i]
        normal = 0 
        if distance < SENSOR_MAX_RANGE:
            normal = 1 - (distance / SENSOR_MAX_RANGE)
        total_sum += normal * vector[i]
    return total_sum

def MotorLogic(readings):    
    
    v_left = BASE_SPEED + K_CONSTANT * BraitSum(readings[0:8], leftBrait)
    v_right = BASE_SPEED + K_CONSTANT * BraitSum(readings[8:16], rightBrait)
    return v_left, v_right

def print_dashboard(readings, state, v_left, v_right):
   
    print(f"\n[{state}] -> vS: {v_left:+.2f} | vD: {v_right:+.2f}")

def main():
    client = RemoteAPIClient()
    sim = client.require('sim')
    robot = sim.getObject('/PioneerP3DX')
    left_motor  = sim.getObject('/PioneerP3DX/leftMotor')
    right_motor = sim.getObject('/PioneerP3DX/rightMotor')

    sensors = [sim.getObject(f'/PioneerP3DX/ultrasonicSensor[{i}]') for i in range(16)]

    sim.startSimulation()
    print(f"Robot pornit! Tinta: {TARGET_DIST}m de peretele drept. (Ctrl+C pentru oprire)\n")

    path_x = []
    path_y = []
    try:
        while True:

            pos = sim.getObjectPosition(robot, sim.handle_world)
            path_x.append(pos[0])
            path_y.append(pos[1])

            readings = read_all_sensors(sim, sensors)
            
            # 1. Extragem distantele minime pe directii
            dist_fata = get_group_min_distance(readings, SENSORS_CENTER)
            dist_stanga = get_group_min_distance(readings, SENSORS_LEFT)
            dist_dreapta = get_group_min_distance(readings, SENSORS_RIGHT)
            dist_spate = get_group_min_distance(readings, SENSORS_BACK)

            state_desc = ""

            

           
            if dist_fata < 0.3 and dist_stanga < 0.4 and dist_dreapta < 0.4:
                optiuni = {"STANGA": dist_stanga, "DREAPTA": dist_dreapta, "SPATE": dist_spate}
                cea_mai_buna_directie = max(optiuni, key=optiuni.get)
                state_desc = f"EVADARE ({cea_mai_buna_directie})"
                
                if cea_mai_buna_directie == "STANGA":
                    v_left, v_right = -BASE_SPEED, BASE_SPEED
                elif cea_mai_buna_directie == "DREAPTA":
                    v_left, v_right = BASE_SPEED, -BASE_SPEED
                else:
                    v_left, v_right = -BASE_SPEED*1.5, BASE_SPEED*1.5 # Rotire rapida
                    
           
            elif dist_fata < FRONT_STOP:
                state_desc = f"COLT INTERIOR (Fata: {dist_fata:.2f}m)"
                v_left, v_right = -BASE_SPEED * 0.5, BASE_SPEED

            
            elif dist_dreapta < 0.8:
                error = dist_dreapta - TARGET_DIST
                v_left  = BASE_SPEED + (K_P * error)
                v_right = BASE_SPEED - (K_P * error)
                state_desc = f"URMARIRE PERETE (Err: {error:+.2f}m)"

           
            else:
                v_left, v_right = MotorLogic(readings)
                state_desc = "BRAITENBERG (Mers liber)"

       
            cap = BASE_SPEED * 1.5
            v_left = max(-cap, min(cap, v_left))
            v_right = max(-cap, min(cap, v_right))

            sim.setJointTargetVelocity(left_motor, v_left)
            sim.setJointTargetVelocity(right_motor, v_right)

            print_dashboard(readings, state_desc, v_left, v_right)
            
            time.sleep(0.05)   

    except KeyboardInterrupt:
        print("\nSimulare oprita manual.")
    finally:
        sim.setJointTargetVelocity(left_motor, 0.0)
        sim.setJointTargetVelocity(right_motor, 0.0)
        sim.stopSimulation()
        if path_x and path_y:
            plt.figure(figsize=(8, 8))
            
            
            plt.plot(path_x, path_y, 'b-', linewidth=2, label="Traseu Robot")
            
            # Mark the start and end points
            plt.plot(path_x[0], path_y[0], 'go', markersize=8, label="Start")
            plt.plot(path_x[-1], path_y[-1], 'rx', markersize=8, label="Stop")
            
          
            plt.title("Traseul Robotului Pioneer P3DX", fontsize=14)
            plt.xlabel("Coordonata X (metri)", fontsize=12)
            plt.ylabel("Coordonata Y (metri)", fontsize=12)
            plt.axis('equal')
            plt.grid(True)
            plt.legend()
            
         
            plt.show()


        

if __name__ == '__main__':
    main()