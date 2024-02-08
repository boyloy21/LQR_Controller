import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import pathlib
from kinematic_mecanum import mecanum
from lqr_controller import LQR_ControllerV2

# from utils.angle import angle_mod
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
import cubic_spline_planner
# from PathPlanning.CubicSpline import cubic_spline_planner

# === Parameters =====

# LQR parameter
lqr_Q = 100*np.eye(6)
lqr_R = 0.01*np.eye(6)
dt = 0.1  # time tick[s]
L = 0.5  # Wheel base of the vehicle [m]
max_steer = np.deg2rad(45.0)  # maximum steering angle[rad]
lx = 0.165
ly = 0.225 
R = 0.076
show_animation = True


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, Vx=0.0, Vy=0.0,Omega=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.Vx = Vx
        self.Vy = Vy
        self.Omega = Omega

def update(state, ax,ay,ayaw,W1,W2,W3,W4):

    if state.yaw >= max_steer:
        state.yaw = max_steer
    if state.yaw <= - max_steer:
        state.yaw = - max_steer
    mec = mecanum(R,lx,ly)
    Vx,Vy,Omega = mec.forward_kinematic(W1,W2,W3,W4)
    # state.x = state.x + Vx*np.cos(state.yaw)*dt - Vy*np.sin(state.yaw)*dt
    # state.y = state.y + Vx*np.sin(state.yaw)*dt - Vy*np.cos(state.yaw)*dt
    state.x = state.x + Vx*dt 
    state.y = state.y + Vy*dt 
    state.yaw = state.yaw + Omega * dt
    state.Vx = state.Vx + ax * dt
    state.Vy = state.Vy + ay * dt
    state.Omega = state.Omega + ayaw * dt

    return state

def angle_mod(x, zero_2_2pi=False, degree=False):
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle
def pi_2_pi(angle):
    return angle_mod(angle)

def calc_nearest_index(state, cx, cy, cyaw):
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind
def lqr_speed_steering_control(state,cx,cy,cyaw,ck,spx,spy,spyaw,dt,Q,R):
    ind,_= calc_nearest_index(state,cx,cy,cyaw)
    k = ck[ind]
    A = np.diag([1,1,1,1,1,1])
    B = np.diag([dt,dt,dt,dt,dt,dt])
    # B = np.zeros((6,6))
    # B[0,0] = np.cos(cyaw)*dt
    # B[0,1] = -np.sin(cyaw)*dt
    # B[1,0] = np.sin(cyaw)*dt
    # B[1,1] = np.cos(cyaw)*dt
    # B[2,2] = dt
    # B[3,3] = dt
    # B[4,4] = dt
    # B[5,5] = dt
    # B = np.array([[np.cos(cyaw)*dt,-np.sin(cyaw)*dt,0.0,0.0,0.0,0.0],
    #              [np.sin(cyaw)*dt, np.cos(cyaw)*dt ,0.0,0.0,0.0,0.0],
    #              [0.0, 0.0, dt, 0.0, 0.0, 0.0],
    #              [0.0, 0.0, 0.0, dt, 0.0, 0.0],
    #              [0.0, 0.0, 0.0, 0.0, dt, 0.0],
    #              [0.0, 0.0, 0.0, 0.0, 0.0, dt]])
    errorx = (state.x - cx[ind])
    errory = (state.y - cy[ind])
    erroryaw = (state.yaw - cyaw[ind])
    errorvx = (state.Vx - spx[ind])
    errorvy = (state.Vy - spy[ind])
    errorvyaw = (state.Omega - spyaw[ind])
    error = np.zeros((6, 1))
    error[0,0] = errorx
    error[1,0] = errory
    error[2,0] = erroryaw
    error[3,0] = errorvx
    error[4,0] = errorvy
    error[5,0] = errorvyaw
    LQR = LQR_ControllerV2(A,B,Q,R,150)
    K,_,_ = LQR.dlqr()
    U = -K @ error

    # # calc steering input
    # ff = math.atan2(L * k, 1)  # feedforward steering angle
    # fb = pi_2_pi(U[2, 0])  # feedback steering angle
    # delta = ff + fb
    return U, ind
def do_simulation(cx, cy, cyaw, ck, spx,spy,spyaw, goal):
    T = 500.0  # max simulation time
    goal_dis = 0.3
    stop_speed = 0.05
    mec = mecanum(R,lx,ly)
    state = State(x=-0.0, y=-0.0, yaw=0.0, Vx=0.0, Vy=0.0,Omega= 0.0)

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    Vx = [state.Vx]
    Vy = [state.Vy]
    Omega = [state.Omega]
    t = [0.0]

    

    while T >= time:
        U, target_ind = lqr_speed_steering_control(
            state, cx, cy, cyaw, ck, spx,spy,spyaw,dt, lqr_Q, lqr_R)
        w1,w2,w3,w4 = mec.inverse_kinematic(U[0,0],U[1,0],U[2,0])
        state = update(state, U[3,0],U[4,0],U[5,0],w1,w2,w3,w4)

        if abs(state.Vx) <= stop_speed:
            target_ind += 1

        time = time + dt

        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if math.hypot(dx, dy) <= goal_dis:
            print("Goal")
            break

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        Vx.append(state.Vx)
        Vy.append(state.Vy)
        Omega.append(state.Omega)
        t.append(time)

        if target_ind % 1 == 0 and show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("speed[km/h]:" + str(round(state.Vx * 3.6, 2))
                      + ",target index:" + str(target_ind))
            plt.pause(0.0001)

    return t, x, y, yaw, Vx,Vy,Omega   
def calc_speed_profile(cyaw, target_speed):
    speed_profile = [target_speed] * len(cyaw)

    direction = 1.0

    # Set stop point
    for i in range(len(cyaw) - 1):
        dyaw = abs(cyaw[i + 1] - cyaw[i])
        Vyaw = dyaw/dt
        switch = math.pi / 4.0 <= dyaw < math.pi / 2.0

        if switch:
            direction *= -1

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

        if switch:
            speed_profile[i] = 0.0

    # speed down
    for i in range(40):
        speed_profile[-i] = target_speed / (50 - i)
        if speed_profile[-i] <= 1.0 / 3.6:
            speed_profile[-i] = 1.0 / 3.6

    return speed_profile
ax = [0.0, 2.0,4.0,5.5, 6.2, 8.5, 9.0]
ay = [0.0, 0.0,-0.0,-0.5, -3.2, -3.0, -1.0]
goal = [ax[-1], ay[-1]]

cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.1)
target_speed = 10.0 / 3.6 

def main():
    print("LQR steering control tracking start!!")
    ax = [0.0, 2.0,4.0,5.5, 6.2, 8.5, 9.0]
    ay = [0.0, 0.0,-0.0,-0.5, -3.2, -3.0, -1.0]
    goal = [ax[-1], ay[-1]]

    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.1)
    

    target_speedx = 10.0 / 3.6  # simulation parameter km/h -> m/s
    target_speedy = 10.0 / 3.6
    target_speedyaw = 3.14
    spx = calc_speed_profile(cyaw, target_speedx)
    spy = calc_speed_profile(cyaw, target_speedy)
    spyaw = calc_speed_profile(cyaw, target_speedyaw)
    
    t, x, y, yaw, v = do_simulation(cx, cy, cyaw, ck, spx,spy,spyaw, goal)

    if show_animation:  # pragma: no cover
        plt.close()
        plt.subplots(1)
        plt.plot(ax, ay, "xb", label="waypoints")
        plt.plot(cx, cy, "-r", label="target course")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots(1)
        plt.plot(s, [np.rad2deg(iyaw) for iyaw in cyaw], "-r", label="yaw")
        plt.grid(True)
        plt.legend()
        plt.xlabel("line length[m]")
        plt.ylabel("yaw angle[deg]")

        plt.subplots(1)
        plt.plot(s, ck, "-r", label="curvature")
        plt.grid(True)
        plt.legend()
        plt.xlabel("line length[m]")
        plt.ylabel("curvature [1/m]")

        plt.show()


if __name__ == '__main__':
    main()