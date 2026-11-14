
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sympy
import pandas as pd
from sympy import symbols, Matrix, latex
sympy.init_printing(use_latex='mathjax')
from scipy.stats import norm

arrow = u'$\u2191$'

def wrap_angle(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi 

# def compute_motion_jacobian(self,u,state,time):
#     x, y, theta, v, w, dt = symbols('x y theta v w dt')
#     R = v / w
#     beta = theta + w * dt
#     gux = Matrix(
#         [
#             [x - R * sympy.sin(theta) + R * sympy.sin(beta)],
#             [y + R * sympy.cos(theta) - R * sympy.cos(beta)],
#             [beta],
#         ]
#     )

#     Gt = gux.jacobian(Matrix([x, y, theta]))
#     Vt = gux.jacobian(Matrix([v, w]))

#     args = (state[0],state[1],state[2],u[0],u[1],time)    
#     eval_gux = sympy.lambdify(args, gux, 'numpy')
#     eval_Gt = sympy.lambdify(args, Gt, "numpy")
#     eval_Vt = sympy.lambdify(args, Vt, "numpy")

#     return Gt, Vt, gux, eval_gux, eval_Gt, eval_Vt

def eval_Gt(u,prev_state,dt): #jacobian with respect to the state (x,y,theta_z or yaw)

         theta=prev_state[2,0] #state stands for (x,y,theta) from the odometry message

         v,w= u[0,0], u[1,0] #velocity command from the odometry message 

         Gt= np.array([[1, 0, (v/w)*(-math.cos(theta)+math.cos(theta+w*dt))],
                       [0,1,(v/w)*(-math.sin(theta)+math.sin(theta+w*dt))],
                       [0,0,1]])
         return Gt

def eval_Vt(u,prev_state,dt): #jacobian with respect to the command

         theta=prev_state[2,0]
         
         v,w= u[0,0], u[1,0]

         Vt=np.array([[(-math.sin(theta)+math.sin(theta+w*dt))/w, (v/(w**2))*(math.sin(theta)-math.sin(theta+w*dt))+(v/w)*(math.cos(theta+w*dt)*dt)],
                      [(1/w)*(math.cos(theta)-math.cos(w*dt)), (-v/(w**2))*(math.cos(theta)-math.cos(theta+w*dt))+(v/w)*(math.sin(theta+w*dt)*dt)],
                      [0,dt]])

         return Vt

def eval_Mt(u,alpha):

         v,w= u[0,0], u[1,0]

         Mt=np.array([[alpha[0]*(v**2)+alpha[1]*(w**2),0],
                      [0,alpha[2]*(v**2)+alpha[3]*(w**2)]]) #define the alpha matrix
         return Mt

def eval_mu(u,prev_mu,dt): # evaluate the new predicted mean value, does the same thing as the velocity model 

         theta=prev_mu[2,0]
         v,w_act= u[0,0], u[1,0]
         
         w= max(w_act,1e-9) 
              
         eval_mu= prev_mu + np.array([[(-v/w)*math.sin(theta)+(v/w)*math.sin(theta+w*dt)],
                                    [(v/w)*math.cos(theta)+(-v/w)*math.cos(theta+w*dt)],
                                    [w*dt]])
         return eval_mu

def Qt(measurement_noise):
    """Construct the measurement noise covariance matrix Q."""
    return np.diag([measurement_noise[0] ** 2, measurement_noise[1] ** 2])

def Ht(pred_mu, landmark_coordinates):

    row=[]
    columns=[]

    x, y, theta = pred_mu[0], pred_mu[1], pred_mu[2]
    for id in landmark_coordinates:
        columns.append(id)
        x_obs, y_obs = landmark_coordinates[id]
        ht=np.array([(x-x_obs)/math.sqrt((-x+x_obs)**2+(-y+y_obs)**2),(y-y_obs)/math.sqrt((-x+x_obs)**2+(-y+y_obs)**2),0],
                    [-(y-y_obs)/math.sqrt((-x+x_obs)**2+(-y+y_obs)**2),-(x-x_obs)/math.sqrt((-x+x_obs)**2+(-y+y_obs)**2),-1])
        row.append(ht)

        Ht=pd.DataFrame(row,columns)

    
    return Ht

#def compute_measurement_jacobian():
    # x, y, theta = symbols('x y theta')
    # mx, my = symbols("m_x m_y")
    # hx = Matrix(
    #     [
    #         [sympy.sqrt((mx - x) ** 2 + (my - y) ** 2)],
    #         [sympy.atan2(my - y, mx - x) - theta],
    #     ]
    # )

    # Ht = hx.jacobian(Matrix([x, y, theta]))
    # eval_hx = sympy.lambdify((x, y, theta, mx, my), hx, "numpy")
    # eval_Ht = sympy.lambdify((x, y, theta, mx, my), Ht, "numpy")    

    # return Ht, hx, eval_hx, eval_Ht

def evaluate_sampling_dist(mu, sigma, n_samples, sample_function, n_bins=100, title="plot"):
    """ 
    Evaluate the sampling distribution by generating samples, 
    computing mean and std dev, and plotting a histogram. 
    """
    samples = []

    for i in range(n_samples):
        samples.append(sample_function(mu, sigma))

    print("%30s : mean = %.3f, std_dev = %.3f" % ("Normal", numpy.mean(samples), numpy.std(samples)))

    plt.figure()
    plt.hist(samples, n_bins)
    plt.xlim([mu - 5*sigma, mu + 5*sigma])
    plt.title("Normal distribution of samples")
    plt.grid()
    plt.savefig(title + ".png")
    return plt

def plot_sampled_model(x, x_sampled, title="", n_plot=500):
    plt.figure()

    # Plot sampled data in red    
    for x_ in x_sampled[:n_plot]:
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x_[2])-90)
        plt.scatter(x_[0], x_[1], marker=rotated_marker, s=40, facecolors='none', edgecolors='r')

    # Plot robot initial pose in blue
    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x[2])-90)
    plt.scatter(x[0], x[1], marker=rotated_marker, s=100, facecolors='none', edgecolors='b')

    # Append initial pose to same view as samples
    """
    xs = numpy.append(x_sampled[:, 0], x[0])
    ys = numpy.append(x_sampled[:, 1], x[1])
    pad = 0.08 * max(numpy.ptp(xs), numpy.ptp(ys))
    plt.xlim(xs.min() - pad, xs.max() + pad)
    plt.ylim(ys.min() - pad, ys.max() + pad)
    plt.gca().set_aspect('equal', adjustable='box')
    """

    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title(title)
    plt.savefig(title + '.png')
    return plt

def compute_p_hit_dist(dist, max_dist, sigma):
    """
    Compute the hit probability p_hit for a given distance measurement using scipy.
    Args:
        dist: observed distance measurement
        max_dist: maximum measurable distance
        sigma: standard deviation of the Gaussian noise
    Returns:
        p_hit: normalized hit probability
    """

    # Normalize the Gaussian PDF with the CDF probability computed over [0, max_dist]
    normalize_hit = norm.cdf(max_dist, loc=0, scale=sigma) - norm.cdf(0, loc=0, scale=sigma)
    normalize_hit = 1. / normalize_hit

    p_hit = norm.pdf(dist, loc=0, scale=sigma) * normalize_hit
    p_hit = numpy.where((dist > max_dist) | (dist < 0.0), 0.0, p_hit)  # Set p_hit to 0 for distances greater than max_dist or negative anomalies

    return p_hit

