import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import pathlib
from kinematic_mecanum import mecanum
from lqr_controller import LQR_ControllerV2,LQR_Controller
from bezier_path import calc_4points_bezier_path

# from utils.angle import angle_mod
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
import cubic_spline_planner

max_linear_velocity = 3.0 # meters per second
max_angular_velocity = 1.5708 # radians per second
lx = 0.165
ly = 0.225 
dt = 0.01
R = 0.076
sim_time = 350
start_x = 0.0
start_y = 0.0
start_yaw = 0
end_x = 6.0
end_y = -3.4
end_yaw = -1.57
offset=1.6
index=0



def getB(yaw, deltat):
    
    B = np.array([  [np.cos(yaw)*deltat, -np.sin(yaw)*deltat, 0],
                                    [np.sin(yaw)*deltat, np.cos(yaw)*deltat, 0],
                                    [0,0, deltat]])
    return B

def state_space_model(A, state_t_minus_1, B, control_input_t_minus_1):

    control_input_t_minus_1[0,0] = np.clip(control_input_t_minus_1[0,0],-max_linear_velocity,max_linear_velocity)
    control_input_t_minus_1[1,0] = np.clip(control_input_t_minus_1[1,0],-max_linear_velocity,max_linear_velocity)
    control_input_t_minus_1[2,0] = np.clip(control_input_t_minus_1[2,0],-max_angular_velocity,max_angular_velocity)
    state = A @ state_t_minus_1 + B @ control_input_t_minus_1

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
    dx = [state[0,0] - icx for icx in cx]
    dy = [state[1,0] - icy for icy in cy]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)

    mind = math.sqrt(mind)

    dxl = cx[ind] - state[0,0]
    dyl = cy[ind] - state[1,0]

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind
def plot_arrow(x, y, yaw, length=0.025, width=0.3, fc="b", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
            fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def main():
    gain_Q = np.eye(3)
    gain_Q[0,0] = 54
    gain_Q[1,1] = 23
    gain_Q[2,2] = 40
    gain_R = 0.08*np.eye(3)
    gain_R[1,1] = 0.08
    gain_R[2,2] = 0.3
    # gain_Q = np.eye(3)
    # gain_Q[0,0] = 65
    # gain_Q[1,1] = 20
    # gain_Q[2,2] = 35
    # gain_R = 0.25*np.eye(3)
    # gain_R[1,1] = 0.2
    # gain_R[2,2] = 0.25
    A = np.diag([1,1,1])
    
    mec = mecanum(R,lx,ly)
    actual_state = np.zeros((3,1))
    # cx = 5
    # cy = 0.0
    # cyaw = 0.0
    #calculate bezier_path
    path, _ = calc_4points_bezier_path(
        start_x, start_y, start_yaw,
        end_x, end_y, end_yaw,
        offset
    )
    index = 0
    ax = path[:, 0]
    ay = path[:, 1]
    ayaw = np.append(np.arctan2(np.diff(ay), np.diff(ax)), end_yaw)
    ref_path = np.vstack([path[:, 0], path[:, 1], np.append(np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0])), end_yaw)])
    ax = [0.0, 2.0,4.0,5.5, 6.2, 8.5, 9.0]
    ay = [0.0, 0.0,-0.0,-0.5, -3.2, -3.0, -1.0]
    goal = [ax[-1], ay[-1]]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)
    ind,_ = calc_nearest_index(actual_state,cx,cy,cyaw)
    errorx = [actual_state[0,0]-ref_path[0, 0]]
    errory = [actual_state[1,0]-ref_path[1, 0]]
    erroryaw = [actual_state[2,0]-ref_path[2, 0]]
    hist_x = [actual_state[0,0]]
    hist_y = [actual_state[1,0]]
    hist_yaw = [actual_state[2,0]]
    for i in range(sim_time):
        if index >=ref_path.shape[1]:
            index=ref_path.shape[1]-1
        errorx.append(actual_state[0,0]-ref_path[0, index])
        errory.append(actual_state[1,0]-ref_path[1, index])
        erroryaw.append(actual_state[2,0]-ref_path[2, index])
        B = getB(actual_state[2,0],dt)
        lqr = LQR_Controller(gain_Q,gain_R,A,B,5)
        U = lqr.Calculate_lqr(errorx[-1],errory[-1],erroryaw[-1])
        if (U[0,0] > 3.0):
            U[0,0] = 3.0
        if (U[0,0] < -3.0):
            U[0,0] = -3.0
        if (U[1,0] > 3.0):
            U[1,0] = 3.0
        if (U[1,0] < -3.0):
            U[1,0] = -3.0
        if (U[2,0] > 1.57):
            U[2,0] = 1.57
        if (U[2,0] < -1.57):
            U[2,0] = -1.57
        print(U[0,0])
        W1,W2,W3,W4 = mec.inverse_kinematic(U[0,0],U[1,0],U[2,0])
        Vx,Vy,Omega = mec.forward_kinematic(W1,W2,W3,W4)
        # V= mec.Rotation_matrix(actual_state[2,0],Vx,Vy,Omega) 
        # Vxr = V[0,0]
        # Vyr = V[1,0]
        # Omegar = V[2,0]

        Input = np.array([[Vx],[Vy],[Omega]])
        actual_state = state_space_model(A,actual_state,B,Input)
        hist_x.append(actual_state[0,0])
        hist_y.append(actual_state[1,0])

        plt.clf()
        plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
        plot_arrow(actual_state[0,0], actual_state[1,0], actual_state[2,0])
        plt.plot(ref_path[0],ref_path[1], marker="x", color="blue", label="Input Trajectory")
        # plt.plot(cx, cy, "-r", label="course")
        plt.plot(hist_x, hist_y, color="red", label="LQR Track")
        plt.title("Velocity of robot [m/sec]:" + str(round(Vx, 2)))
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.pause(0.0001)
        index+=1

if __name__ == '__main__':
    main()
    


