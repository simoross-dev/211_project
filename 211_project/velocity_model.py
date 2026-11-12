import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, degrees
import matplotlib as mpl

import sympy
sympy.init_printing(use_latex='mathjax')
from localization_utils import compute_motion_jacobian, evaluate_sampling_dist, plot_sampled_model

class VelocityModel:
    def __init__(
            self,
            x=None,
            dim_x=None,
            u=None,
            a=None,
            dt=None,
            n_samples=None
        ):

        self.x_sampled = np.zeros([n_samples, dim_x])
        for i in range(n_samples):
            self.x_sampled[i,:] = self.sample_velocity_motion_model(x, u, a, dt)

        self.Gt, self.Vt, self.gux, self.eval_gux, self.eval_Gt, self.eval_Vt = compute_motion_jacobian()

    def sample_velocity_motion_model(self, x, u, a, dt):
        """ 
        Sample velocity motion model.
        Arguments:
        x -- pose of the robot before moving [x, y, theta]
        u -- velocity reading obtained from the robot [v, w]
        a -- noise parameters of the motion model [a1, a2, a3, a4, a5, a6] 
                where a1, a3, a5 are related to v,
                and a2, a4, a6 to w
        """
        v_hat = u[0] + np.random.normal(0, a[0]*u[0]**2 + a[1]*u[1]**2)
        w_hat = u[1] + np.random.normal(0, a[2]*u[0]**2 + a[3]*u[1]**2)
        gamma_hat = np.random.normal(0, a[4]*u[0]**2 + a[5]*u[1]**2)

        if abs(w_hat) < 1e-9:
            # straight line
            x_prime = x[0] + v_hat*dt*cos(x[2])
            y_prime = x[1] + v_hat*dt*sin(x[2])
        else:
            r = v_hat/w_hat
            x_prime = x[0] - r*sin(x[2]) + r*sin(x[2]+w_hat*dt)
            y_prime = x[1] + r*cos(x[2]) - r*cos(x[2]+w_hat*dt)
        theta_prime = x[2] + w_hat*dt + gamma_hat*dt

        return np.array([x_prime, y_prime, theta_prime])


def main():
    n_samples = 500 # number of samples
    dt = 0.5

    x = [0, 0, 0] # initial pose
    u = [0.2, 0.8] # velocity command as per Turtlebot3's specs (linear and angular)

    noise_v = [0.3, 0, 0.2, 0, 0.5, 0] # noise variance for linear uncertainty
    noise_w = [0, 0.4, 0, 0.4, 0, 0.6]  # noise variance for angular uncertainty

    # Scenario 1 - high uncertainty on linear motion
    velModelLinear = VelocityModel(x, len(x), u, noise_v, dt, n_samples)

    x_prime_linear_noise = np.zeros([n_samples, 3])
    for i in range(n_samples):
        x_prime_linear_noise[i,:] = velModelLinear.sample_velocity_motion_model(x, u, noise_v, dt)

    # Scenario 2 - high uncertainty on angular motion
    velModelAngular = VelocityModel(x, len(x), u, noise_w, dt, n_samples)
    
    x_prime_angular_noise = np.zeros([n_samples, 3])
    for i in range(n_samples):
        x_prime_angular_noise[i,:] = velModelAngular.sample_velocity_motion_model(x, u, noise_w, dt)

    # Sampling the velocity model with high linear uncertainty  
    mu = np.mean(x_prime_linear_noise, axis=0)
    sigma = np.std(x_prime_linear_noise, axis=0)
    plt1 = evaluate_sampling_dist(mu[0], sigma[0], n_samples, np.random.normal, title="velocity_model_gaussian_dist_noise_v")
    plt1.show()

    # Sampling the velocity model with high angular uncertainty  
    mu = np.mean(x_prime_angular_noise, axis=0)
    sigma = np.std(x_prime_angular_noise, axis=0)
    plt2 = evaluate_sampling_dist(mu[0], sigma[0], n_samples, np.random.normal, title="velocity_model_gaussian_dist_noise_w")
    plt2.show()

    # Plot the two samples
    plt3 = plot_sampled_model(x, x_sampled=x_prime_linear_noise, title="velocity motion model sampling (high linear noise)")
    plt3.show()

    plt4 = plot_sampled_model(x, x_sampled=x_prime_angular_noise, title="velocity motion model sampling (high angular noise)")
    plt4.show()

    # Scenario 3 - high uncertainty on v and w    
    noise = [0.6, 0.7, 0.8, 0.5, 0.5, 0.7]
    velModel = VelocityModel(x, len(x), u, noise, dt, n_samples)

    x_prime = np.zeros([n_samples, 3])
    for i in range(n_samples):
        x_prime[i,:] = velModel.sample_velocity_motion_model(x, u, noise, dt)
    
    mu = np.mean(x_prime, axis=0)
    sigma = np.std(x_prime, axis=0)

    plt5 = evaluate_sampling_dist(mu[0], sigma[0], n_samples, np.random.normal, title="velocity_model_noisy_gaussian_dist")
    plt5.show()

    plt6 = plot_sampled_model(x, x_sampled=x_prime, title="velocity motion model sampling (high noise)")
    plt6.show()

    # Compute Jacobian of non-linear model wrt x, and u
    Gt, Vt, gux, eval_gux, eval_Gt, eval_Vt = compute_motion_jacobian()
    print("Jacobian for Non-linear model wrt to the state Gt:")
    print(sympy.simplify(Gt))
    print("Jacobian for Non-linear model wrt to the command variable Vt:")
    print(sympy.simplify(Vt))

    # Evalute computed jacobians
    G_num = np.array(eval_Gt(*x, *u, dt), dtype=float)
    print(sympy.simplify(G_num))
    V_num = np.array(eval_Vt(*x, *u, dt), dtype=float)
    print(sympy.simplify(V_num))

    plt.close('all')

if __name__ == "__main__":
    main()