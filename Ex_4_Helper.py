import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

g = 9.8
theta_israel = (np.pi / 6)
omega_tot = 2 * np.pi / (24 * 3600)  # omega of earth
omega_x = 0
omega_y = omega_tot * np.cos(theta_israel)
omega_z = omega_tot * np.sin(theta_israel)
alpha = 0.05
R_base = 200


def acceleration(v_x, v_y, v_z, m, A):
    a_x = - (alpha * A * np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2) * v_x / m + 2 * (omega_y * v_z - v_y * omega_z))
    a_y = - (alpha * A * np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2) * v_y / m + 2 * (v_x * omega_z - omega_x * v_z))
    a_z = - (g + (alpha * A * np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2) * v_z / m) + 2 * (v_y * omega_x - omega_y * v_x))
    return a_x, a_y, a_z

def runge_kutta_rocket(r_0=(0, 0, 0), dt=0.01, N=10000, m=500,v_0=(500 * np.cos(np.radians(20)) * np.sin(np.radians(-90)),
                            500 * np.cos(np.radians(-90)) * np.cos(np.radians(20)),
                            500 * np.sin(np.radians(20))), A=1, sensitivity=0.01):
    t = np.linspace(start=0, num=N, stop=0 + (N - 1) * dt)
    x = np.ndarray(N)
    y = np.ndarray(N)
    z = np.ndarray(N)
    vx = np.ndarray(N)
    vy = np.ndarray(N)
    vz = np.ndarray(N)

    x[0], y[0], z[0] = r_0
    vx[0], vy[0], vz[0] = v_0

    for i in range(N - 1):
        x[i + 1], y[i + 1], z[i + 1], vx[i + 1], vy[i + 1], vz[i + 1] = runga_step(x[i], y[i], z[i], vx[i], vy[i],
                                                                                   vz[i], m, A, dt)

        if (z[i + 1] < 0):
            break

    i_ground = np.argmax(z < 0.)

    t[i_ground], x[i_ground], y[i_ground], z[i_ground] = final_step_to_ground_no_randomness(
        (x[i_ground - 1], y[i_ground - 1], z[i_ground - 1]), (x[i_ground - 2], y[i_ground - 2], z[i_ground - 2]),
        (vx[i_ground - 1], vy[i_ground - 1], vz[i_ground - 1]), sensitivity, m, A, t[i_ground - 1], t[i_ground - 2])
    return t[:i_ground + 1], (x[:i_ground + 1], y[:i_ground + 1], z[:i_ground + 1]), i_ground


def runga_step(x, y, z, v_x, v_y, v_z, m, A, dt):
    k1_x, k1_y, k1_z = v_x, v_y, v_z
    k1_vx, k1_vy, k1_vz = acceleration(v_x, v_y, v_z, m, A)

    k2_x, k2_y, k2_z = (v_x + (dt / 2) * k1_vx), (v_y + (dt / 2) * k1_vy), (v_z + (dt / 2) * k1_vz)
    k2_vx, k2_vy, k2_vz = acceleration((v_x + (dt / 2) * k1_vx), (v_y + (dt / 2) * k1_vy),
                                       (v_z + (dt / 2) * k1_vz), m, A)

    k3_x, k3_y, k3_z = (v_x + (dt / 2) * k2_vx), (v_y + (dt / 2) * k2_vy), (v_z + (dt / 2) * k2_vz)
    k3_vx, k3_vy, k3_vz = acceleration((v_x + (dt / 2) * k2_vx), (v_y + (dt / 2) * k2_vy),
                                       (v_z + (dt / 2) * k2_vz), m, A)

    k4_x, k4_y, k4_z = (v_x + dt * k3_vx), (v_y + dt * k3_vy), (v_z + dt * k3_vz)
    k4_vx, k4_vy, k4_vz = acceleration((v_x + dt * k3_vx), (v_y + dt * k3_vy), (v_z + dt * k3_vz), m, A)

    # recurssion step
    x_return, y_return, z_return = (x + 1 / 6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x) * dt), (
            y + 1 / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y) * dt), (
                                           z + 1 / 6 * (k1_z + 2 * k2_z + 2 * k3_z + k4_z) * dt)

    vx_return, vy_return, vz_return = (v_x + 1 / 6 * (k1_vx + 2 * k2_vx + 2 * k3_vx + k4_vx) * dt), (
            v_y + 1 / 6 * (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy) * dt), (
                                              v_z + 1 / 6 * (k1_vz + 2 * k2_vz + 2 * k3_vz + k4_vz) * dt)

    return x_return, y_return, z_return, vx_return, vy_return, vz_return


def final_step_to_ground_no_randomness(r_curr, r_prev, v_curr, sensitivity, m, A, t_curr, t_prev):
    r = np.array(r_curr)
    r_old = np.array(r_prev)
    vx, vy, vz = v_curr
    t = t_curr
    t_old = t_prev
    while abs(r[2]) > sensitivity:
        dt = - r[2] * (t - t_old) / (r[2] - r_old[2])
        t_old = t
        t += dt
        r_old = np.array([r[0], r[1], r[2]])
        r[0], r[1], r[2], vx, vy, vz = runga_step(r[0], r[1], r[2], vx, vy, vz, m, A, dt)
    return t, r[0], r[1], r[2]


def acceleration_with_random_and_with_correction(v_x, v_y, v_z, m, A, dt, correction, beta=1000):
    theta, phi = get_angles_from_v((v_x, v_y, v_z))
    a_correction_z = (correction[1] * np.cos(theta)) / m
    a_correction_x = (- correction[1] * np.sin(theta) * np.sin(phi) + correction[0] * np.cos(phi)) / m
    a_correction_y = (- correction[1] * np.sin(theta) * np.cos(phi) - correction[0] * np.sin(phi)) / m
    fr_x, fr_y, fr_z = random_vector()
    inverse_dt = 1 / np.sqrt(dt / 4)
    fr_x, fr_y, fr_z = fr_x * beta * A * inverse_dt / m, fr_y * beta * A * inverse_dt / m, fr_z * beta * A * inverse_dt / m
    a_x = fr_x + a_correction_x - (
            alpha * A * np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2) * v_x / m + 2 * (omega_y * v_z - v_y * omega_z))
    a_y = fr_y + a_correction_y - (
            alpha * A * np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2) * v_y / m + 2 * (v_x * omega_z - omega_x * v_z))
    a_z = fr_z + a_correction_z - (g + (alpha * A * np.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2) * v_z / m) + 2 * (
            v_y * omega_x - omega_y * v_x))
    return a_x, a_y, a_z


def random_vector():
    return (np.random.normal(), np.random.normal(), np.random.normal())


def runga_step_with_correction(x, y, z, v_x, v_y, v_z, m, A, dt, correction, beta):
    k1_x, k1_y, k1_z = v_x, v_y, v_z
    k1_vx, k1_vy, k1_vz = acceleration_with_random_and_with_correction(v_x, v_y, v_z, m, A, dt, correction, beta=beta)

    k2_x, k2_y, k2_z = (v_x + (dt / 2) * k1_vx), (v_y + (dt / 2) * k1_vy), (v_z + (dt / 2) * k1_vz)
    k2_vx, k2_vy, k2_vz = acceleration_with_random_and_with_correction((v_x + (dt / 2) * k1_vx),
                                                                       (v_y + (dt / 2) * k1_vy),
                                                                       (v_z + (dt / 2) * k1_vz), m, A, dt, correction,
                                                                       beta=beta)

    k3_x, k3_y, k3_z = (v_x + (dt / 2) * k2_vx), (v_y + (dt / 2) * k2_vy), (v_z + (dt / 2) * k2_vz)
    k3_vx, k3_vy, k3_vz = acceleration_with_random_and_with_correction((v_x + (dt / 2) * k2_vx),
                                                                       (v_y + (dt / 2) * k2_vy),
                                                                       (v_z + (dt / 2) * k2_vz), m, A, dt, correction,
                                                                       beta=beta)

    k4_x, k4_y, k4_z = (v_x + dt * k3_vx), (v_y + dt * k3_vy), (v_z + dt * k3_vz)
    k4_vx, k4_vy, k4_vz = acceleration_with_random_and_with_correction((v_x + dt * k3_vx), (v_y + dt * k3_vy),
                                                                       (v_z + dt * k3_vz), m, A, dt, correction,
                                                                       beta=beta)

    # recurssion step
    x_return, y_return, z_return = (x + 1 / 6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x) * dt), (
            y + 1 / 6 * (k1_y + 2 * k2_y + 2 * k3_y + k4_y) * dt), (
                                           z + 1 / 6 * (k1_z + 2 * k2_z + 2 * k3_z + k4_z) * dt)

    vx_return, vy_return, vz_return = (v_x + 1 / 6 * (k1_vx + 2 * k2_vx + 2 * k3_vx + k4_vx) * dt), (
            v_y + 1 / 6 * (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy) * dt), (
                                              v_z + 1 / 6 * (k1_vz + 2 * k2_vz + 2 * k3_vz + k4_vz) * dt)

    return x_return, y_return, z_return, vx_return, vy_return, vz_return


def get_angles_from_v(v):
    """
    :param v:
    :return: returns the theta and phi elements of the velocity
    """
    theta = np.arcsin(v[2] / np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2))
    phi = np.arctan2(v[0], v[1])
    return theta, phi




def my_runge_kutta_rocket(r_0=(0, 0, 0), dt=0.01, N=10000, m=500,v_0=(500 * np.cos(np.radians(20)) * np.sin(np.radians(-90)),
                            500 * np.cos(np.radians(-90)) * np.cos(np.radians(20)),
                            500 * np.sin(np.radians(20))), A=1, sensitivity=0.01):
    t = np.linspace(start=0, num=N, stop=0 + (N - 1) * dt)
    x = np.ndarray(N)
    y = np.ndarray(N)
    z = np.ndarray(N)
    vx = np.ndarray(N)
    vy = np.ndarray(N)
    vz = np.ndarray(N)

    x[0], y[0], z[0] = r_0
    vx[0], vy[0], vz[0] = v_0

    for i in range(N - 1):
        x[i + 1], y[i + 1], z[i + 1], vx[i + 1], vy[i + 1], vz[i + 1] = runga_step(x[i], y[i], z[i], vx[i], vy[i],
                                                                                   vz[i], m, A, dt)

        if (z[i + 1] < 0):
            break

    i_ground = np.argmax(z < 0.)

    t[i_ground], x[i_ground], y[i_ground], z[i_ground] = final_step_to_ground_no_randomness(
        (x[i_ground - 1], y[i_ground - 1], z[i_ground - 1]), (x[i_ground - 2], y[i_ground - 2], z[i_ground - 2]),
        (vx[i_ground - 1], vy[i_ground - 1], vz[i_ground - 1]), sensitivity, m, A, t[i_ground - 1], t[i_ground - 2])
    return t[:i_ground + 1], (x[:i_ground + 1], y[:i_ground + 1], z[:i_ground + 1]), i_ground, (vx[:i_ground + 1], vy[:i_ground + 1], vz[:i_ground + 1])

