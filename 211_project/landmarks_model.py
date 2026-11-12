import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from  matplotlib.patches import Arc
from localization_utils import compute_measurement_jacobian, compute_p_hit_dist, wrap_angle, plot_sampled_model

class LandmarkModel:
    def __init__(
            self,
            x=None,
            m=None,
            sigma=None,
            n_samples=None):
        """
        x: robot pose [x, y, theta]
        m: landmark coordinate in the map [m_x, m_y]
        sigma: uncertainty parameters of the sensor [sigma_r, sigma_phi]
        n_samples: number of noisy measurements z to simulate
        """
        self.x = np.asarray(x, dtype=float)
        self.m = np.asarray(m, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.n_samples = n_samples

        if self.x.shape != (3,):
            raise ValueError("x must contain [x, y, theta].")

        if self.sigma.shape != (2,) or np.any(self.sigma < 0):
            raise ValueError("sigma must contain two non-negative values.")

        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
 
        # Symbolic measurement model and Jacobian
        self.Ht, self.hx, self.eval_hx, self.eval_Ht = compute_measurement_jacobian()
 
        # Noise-free measurement for each landmark
        self.z_true = np.array([
            self.expected_measurement(self.x, landmark)
            for landmark in self.m
        ])
 
        # Forward model - generate noisy measurements
        self.z_samples = self.sample_measurements()
 
        # Inverse model - generate candidate robot poses
        self.x_candidates = self.inverse_measurements(self.z_samples, self.m)


    def expected_measurement(self, x, m):
        """Noise-free h(x, m) = [range, bearing] evaluated with the sympy-lambdified model."""
        z = np.array(self.eval_hx(x[0], x[1], x[2], m[0], m[1]), dtype=float).flatten()
        z[1] = wrap_angle(z[1])
        return z

    # Forward model
    def sample_landmark_model(self, x, m, sigma):
        """
        Sampling form h(x, m, sigma).
        x -- robot pose [x, y, theta] 
        m -- landmark [m_x, m_y]
        sigma -- [sigma_r, sigma_phi]
        Returns the estimated range and bearing of the landmark as a numpy array [r, phi].
        """
        z = self.expected_measurement(x, m)
        # Add noise
        z = z + np.random.normal(0.0, sigma, size=2)

        # Bearing should be in [-pi, pi)
        z[1] = wrap_angle(z[1])
        return z    

    def sample_measurements(self):
        """
        Generate n_samples measurements using the forward model.
        """
        z_samples = np.array([
            [
                self.sample_landmark_model(
                    self.x,
                    landmark,
                    self.sigma
                )
                for _ in range(self.n_samples)
            ]
            for landmark in self.m
        ])
        
        return z_samples    

    # Inverse model
    def inverse_landmark_model(self, z, m):
        """
        z : array for measurement [range, bearing].
        m : array for landmark position [m_x, m_y].

        Returns
        -------
        numpy array -  Candidate robot pose [x, y, theta].
        """
        r, phi = z

        # Random angle around the landmark
        gamma = np.random.default_rng(42).uniform(0.0, 2.0 * math.pi)

        # Candidate robot position
        x_candidate = m[0] + r * math.cos(gamma)
        y_candidate = m[1] + r * math.sin(gamma)

        # Candidate robot orientation
        theta_candidate = wrap_angle(
            gamma + math.pi - phi
        )

        return np.array([
            x_candidate,
            y_candidate,
            theta_candidate
        ])

    def inverse_measurements(self, z_samples, m):
        """
        Generate one candidate robot pose for each measurement.

        Parameters
        ----------
        z_samples : array
            Noisy measurements, shape (n_samples, 2).

        m : array
            Landmark position.

        Returns
        -------
        numpy array
            Candidate robot poses, shape (n_samples, 3).
        """

        x_candidates = np.array([
            [
                self.inverse_landmark_model(z, landmark)
                for z in landmark_measurements
            ]
            for landmark, landmark_measurements
            in zip(m, z_samples)
        ])

        return x_candidates

    def compute_jacobian(self, x, m):
        H = np.array([
            self.eval_Ht(
                x[0], x[1], x[2],
                landmark[0], landmark[1]
            )
            for landmark in m
        ], dtype=float)
        return H

    def plot_candidate_poses(self):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot candidate robot poses for each landmark
        for i in range(len(self.m)):

            candidates = self.x_candidates[i]

            ax.scatter(
                candidates[:, 0],
                candidates[:, 1],
                s=12,
                alpha=0.35,
                label=f"Landmark {i+1} samples"
            )

        # Plot true robot position
        ax.scatter(
            self.x[0],
            self.x[1],
            marker="*",
            s=250,
            color="black",
            edgecolors="white",
            label="True robot position",
            zorder=10
        )

        # Plot landmarks
        ax.scatter(
            self.m[:, 0],
            self.m[:, 1],
            marker="s",
            s=120,
            color="green",
            edgecolors="black",
            label="Landmarks",
            zorder=5
        )

        # Label landmarks
        for i, landmark in enumerate(self.m):
            ax.annotate(
                f"L{i+1}",
                (landmark[0], landmark[1]),
                xytext=(7, 7),
                textcoords="offset points"
            )

        ax.set_xlabel("X position [m]")
        ax.set_ylabel("Y position [m]")
        ax.set_title(
            "Sampled Robot Poses - Landmark Measurement Model"
        )

        ax.legend()
        ax.grid(True)
        ax.set_aspect("equal", adjustable="datalim")

        plt.tight_layout()
        plt.show()    

    def plot_measurement_samples(self):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for i in range(len(self.m)):
            color = f"C{i}"

            # Range distributions
            axes[0].hist(
                self.z_samples[i, :, 0],
                bins=40,
                density=True,
                alpha=0.3,
                color=color,
                label=f"Landmark {i+1}"
            )

            # Bearing distributions
            axes[1].hist(
                self.z_samples[i, :, 1],
                bins=40,
                density=True,
                alpha=0.3,
                color=color,
                label=f"Landmark {i+1}"
            )

        axes[0].set_xlabel("Range [m]")
        axes[0].set_ylabel("Probability density")
        axes[0].set_title("Range measurement distributions")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].set_xlabel("Bearing [rad]")
        axes[1].set_ylabel("Probability density")
        axes[1].set_title("Bearing measurement distributions")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

def main():

    # Define LandmarkModel class
    n_samples = 1000

    robot_pose = np.array([0., 0., math.pi/4])
        
    # Landmarks in the map
    landmarks = [
        np.array([5., 2.]),
        np.array([-2.5, 3.]),
        np.array([3., 1.5]),
        np.array([4., -1.]),
        np.array([-2., -2.])
        ]

    # Sensor uncertainty
    sigma = np.array([0.3, math.pi/24])

    model = LandmarkModel(
        x=robot_pose,
        m=landmarks,
        sigma=sigma,
        n_samples=n_samples,
    )

    print("Noise-free measurements [range, bearing]:")
    for i, z in enumerate(model.z_true):
        print(f"Landmark {i+1}: {z}")

    H = model.compute_jacobian(robot_pose, landmarks)
    print("Measurement Jacobian H")
    for i, jacobian in enumerate(H):
        print(f"\nLandmark {i+1}:")
        print(jacobian)
    
    # Plot measurement distributions
    model.plot_measurement_samples()

    # Plot candidate robot poses
    model.plot_candidate_poses()
    
    plt.close('all')

if __name__ == "__main__":
    main()