import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import Ex_4_Helper as helper

g = 9.8
theta_israel = (np.pi / 6)
omega_tot = 2 * np.pi / (24 * 3600)  # omega of earth
omega_x = 0
omega_y = omega_tot * np.cos(theta_israel)
omega_z = omega_tot * np.sin(theta_israel)
alpha = 0.05
R_base = 200


def get_landing_spot(r, v, dt=0.1, N=10000, m=500, A=1, sensitivity=0.01):
    t, r_land, i = helper.runge_kutta_rocket(r_0=r, dt=dt, N=N, m=m, v_0=v, A=A, sensitivity=sensitivity)
    return r_land[0], r_land[1]


def get_angles_from_v(v):
    """
    :param v:
    :return: returns the theta and phi elements of the velocity
    """
    theta = np.arcsin(v[2] / np.sqrt(v[0] ** 2 + v[1] ** 2) + v[2] ** 2)
    phi = np.arctan2(v[0], v[1])
    return theta, phi


def runge_kutta_rocket_with_random_and_correction(r_0=(0, 0, 0), v_0=500, dt=0.1, N=10000, m=500, theta=np.radians(20),
                                                  phi=np.radians(-90), A=1, sensitivity=0.01, beta=1000):
    """
    :param r_0: starting location of the rocket
    :param v_0: starting velocity of the rocket
    :param dt: time interval
    :param N: maximum number of iterations
    :param m: the rocket's mass
    :param theta: angle of launch in respective to the ground
    :param phi: angle of launch in respective to the north
    :param A: the rocket's cross-sectional Area
    :param sensitivity: a parameter of the simulation - leave it as it is.
    :param beta: the coefficient of the random force the rocket gets.
    :return: t - an array of the time,
             r - a 3 tuple of three numpy arrays containing the coordinates of the rocket at each time,
             i - the last index in the arrays - (len(array) - 1)

    """
    t = np.linspace(start=0, num=N, stop=0 + (N - 1) * dt)
    x = np.ndarray(N)
    y = np.ndarray(N)
    z = np.ndarray(N)
    vx = np.ndarray(N)
    vy = np.ndarray(N)
    vz = np.ndarray(N)


    x[0], y[0], z[0] = r_0
    vx[0], vy[0], vz[0] = v_0 * np.sin(phi) * np.cos(theta), v_0 * np.cos(phi) * np.cos(theta), v_0 * np.sin(theta)

    x_tearget, y_target = get_landing_spot((x[0], y[0], z[0]), (vx[0], vy[0], vz[0]), dt=dt, N=N, m=m, A=A,
                                           sensitivity=sensitivity)
    correction = (0, 0)
    theoretical_t, theoretical_r, _,theoretical_v = helper.my_runge_kutta_rocket(dt=delta_t, N=n)
    for i in range(N - 1):
        x[i + 1], y[i + 1], z[i + 1], vx[i + 1], vy[i + 1], vz[i + 1] = helper.runga_step_with_correction(x[i], y[i],
                                                                                                          z[i],
                                                                                                          vx[i], vy[i],
                                                                                                          vz[i], m, A,
                                                                                                          dt,
                                                                                                          correction=correction,
                                                                                                          beta=beta)


        # you may create whatever variables you want in runge_kutta_rocket_with_random_and_correction
        # and have calculate_correction function get whatever parameters you want.

        current_position = (x[i], y[i ], z[i])
        current_velocity = (vx[i ], vy[i ], vz[i ])
        correction = calculate_correction(current_position,current_velocity,theoretical_r,dt, theoretical_v)
        if (z[i + 1] < 0):
            break

    i_ground = np.argmax(z < 0.)

    t[i_ground], x[i_ground], y[i_ground], z[i_ground] = helper.final_step_to_ground_no_randomness(
        (x[i_ground - 1], y[i_ground - 1], z[i_ground - 1]), (x[i_ground - 2], y[i_ground - 2], z[i_ground - 2]),
        (vx[i_ground - 1], vy[i_ground - 1], vz[i_ground - 1]), sensitivity, m, A, t[i_ground - 1], t[i_ground - 2])
    return t[:i_ground + 1], (x[:i_ground + 1], y[:i_ground + 1], z[:i_ground + 1]), i_ground


# create this function so that the location of impact is within 20 meters of the target
Kp = 105
Ki = 0.275
Kd = 2200
# Target position
target_x = -8394.459213691083
target_y = 7.4376783854855395
target_position = np.array([target_x,target_y,0])
target_radius = 10
# Initialize integral and previous error for PID controller
integral_x = 0
integral_y = 0
integral_z = 0
previous_error_x = 0
previous_error_y = 0
previous_error_z = 0
def calculate_correction(current_pos, current_velocity,theoretical_pos, dt =0.1,theoretical_v = 0, target_position=(target_x, target_y)):
    # return (0, 0)
    """
       Calculate the correction forces based on the current position and velocity.

       :param current_position: Current position of the missile
       :param current_velocity: Current velocity of the missile
       :param dt: Time step of the simulation
       :return: Tuple of (horizontal force, vertical force)
       """
    # Calculate errors in x and y directions
    # Calculate the landing spot based on current position and velocity
    global errors
    global integral_x, integral_y, integral_z
    global previous_error_x, previous_error_y, previous_error_z
    global corrections
    x, y, z = current_pos
    vx, vy, vz = current_velocity
    theoretical_positions = np.array(theoretical_pos).T
    theoretical_velocities = np.array(theoretical_v).T
    mask = theoretical_positions[:, 0] <= x
    if not np.any(mask):
        return (0, 0)  # If no such point exists, no correction is applied
    theoretical_positions = theoretical_positions[mask]
    theoretical_velocities = theoretical_velocities[mask]
    # distances = np.linalg.norm(theoretical_positions - np.array([x, y, z]), axis=1)
    distances = np.linalg.norm(theoretical_positions[:, [0, 2]] - np.array([x, z]), axis=1)
    nearest_index = np.argmin(distances)
    x_theoretical, y_theoretical, z_theoretical = theoretical_positions[nearest_index]
    vx_theoretical, vy_theoretical, vz_theoretical = theoretical_velocities[nearest_index]
    error_x = x_theoretical - x
    error_y = y_theoretical - y
    error_z = z_theoretical - z
    # Proportional control
    correction_x = Kp * error_x
    correction_y = Kp * error_y
    correction_z = Kp * error_z

    # Integral term
    integral_x += error_x * dt
    integral_y += error_y * dt
    integral_z += error_z * dt
    I_x = Ki * integral_x
    I_y = Ki * integral_y
    I_z = Ki * integral_z

    # Derivative term
    D_x = Kd * (error_x - previous_error_x) / dt
    D_y = Kd * (error_y - previous_error_y) / dt
    D_z = Kd * (error_z - previous_error_z) / dt

    # Update previous errors
    previous_error_x = error_x
    previous_error_y = error_y
    previous_error_z = error_z

    # Total correction
    correction_x = correction_x + I_x + D_x
    correction_y = correction_y + I_y + D_y
    correction_z = correction_z + I_z + D_z

    theta, phi = helper.get_angles_from_v(current_velocity / np.sqrt(vx**2 + vy**2 + vz**2))

    # Decompose correction forces into engine forces

    # f_0 =0
    f_1 =  np.sign(np.cos(theta)) * correction_z / np.cos(theta)
    f_0 = -(correction_y + f_1 * np.sin(theta)*np.cos(phi))/np.sin(phi)
    # f_1 = correction_z
    f_x = -f_1 * np.sin(theta) * np.sin(phi) + f_0 * np.cos(phi)
    f_y = -f_1 * np.sin(theta) * np.cos(phi) - f_0 * np.sin(phi)
    f_z = f_1 * np.cos(theta)

    # Return the correction tuple (f_0, f_1)
    max_force = 1000000
    f_0 = np.clip(f_0,-max_force,max_force)
    f_1 = np.clip(f_1,-max_force,max_force)
    errors.append(np.sqrt(error_z ** 2) * np.sign(z - z_theoretical))
    corrections.append(f_z)
    return (f_0, f_1)

if __name__ == '__main__':
    n = 10000
    delta_t = 0.01
    t, r, i = helper.runge_kutta_rocket(dt=delta_t, N=n)
    plt.plot(r[0], r[2])
    print(r[0][-1], r[1][-1])
    # plt.show()

    n = 10000
    delta_t = 0.01
    errors = []
    corrections = []
    t, r, i = runge_kutta_rocket_with_random_and_correction(dt=delta_t, N=n)
    plt.plot(r[0], r[2])
    print(r[0][-1], r[1][-1])
    plt.show()


    plt.figure(figsize=(12, 6))
    plt.plot(t[:len(errors)], errors, label='Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Error Throughout the Flight')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(t[:len(corrections)], corrections, label='correction')
    plt.xlabel('Time (s)')
    plt.ylabel('correction (m)')
    plt.title('correction Throughout the Flight')
    plt.legend()
    plt.grid(True)
    plt.show()

    # results = [(runge_kutta_rocket_with_random_and_correction(dt=delta_t, N=n)[1][0][-1] + 8394.459213691083) for _ in
    #            range(50)]
    results = [np.sqrt((res[1][0][-1] + 8394.459213691083) ** 2 + (res[1][1][-1] - 7.437678543884844) ** 2)
               for res in (runge_kutta_rocket_with_random_and_correction(dt=delta_t, N=n) for _ in range(1000))]
    print(np.mean(np.abs(results)))


    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(results, marker='o', linestyle='-', markersize=2)
    plt.title('Landing Error (1000 runs)')
    plt.xlabel('Run')
    plt.ylabel('landing error')
    plt.grid(True)
    plt.show()