import numpy as np
from scipy.integrate import solve_ivp
from scipy import integrate
from scipy.special import genlaguerre, sph_harm, factorial
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for saving
import os
import pandas as pd
import gc  # For garbage collection


class HydrogenWavefunction:
    def __init__(self, n, l, m):
        if n < 1 or l >= n or abs(m) > l:
            raise ValueError("Invalid quantum numbers")
        self.n = n
        self.l = l
        self.m = m
        self.E = -1.0 / (2 * n * n)

    def analytical_radial(self, r):
        rho = 2 * r / self.n
        norm = np.sqrt((2.0 / self.n) ** 3 *
                       factorial(self.n - self.l - 1) / (2 * self.n * factorial(self.n + self.l)))
        L = genlaguerre(self.n - self.l - 1, 2 * self.l + 1)(rho)
        R = norm * np.exp(-rho / 2) * (rho ** self.l) * L
        return R

    def numerical_radial(self, r):
        """
        Solve the radial Schrödinger equation numerically using solve_ivp
        with proper normalization
        """

        def radial_equation(r, y):
            if r < 1e-10:
                r = 1e-10
            V = -1.0 / r
            L_term = self.l * (self.l + 1) / (r * r)
            return [y[1], (2.0 * (V - self.E) + L_term) * y[0]]

        # Initial conditions near r=0
        r0 = 1e-5
        u0 = r0 ** (self.l + 1)
        du0 = (self.l + 1) * r0 ** self.l

        # Solve the differential equation
        solution = solve_ivp(
            radial_equation,
            t_span=(r0, r[-1]),
            y0=[u0, du0],
            t_eval=r,
            method='RK45',
            rtol=1e-12,
            atol=1e-12
        )

        # Convert from u(r) to R(r)
        R = np.where(r > 1e-10, solution.y[0] / r, 0)

        # Calculate normalization integral
        from scipy.integrate import simps
        norm_integral = simps(R * R * r * r, r)

        # Normalize the wavefunction
        R = R / np.sqrt(norm_integral)

        # Ensure consistent sign with analytical solution
        ref_point = self.n * 0.5
        ref_idx = np.abs(r - ref_point).argmin()
        if ref_idx > 0:
            if R[ref_idx] * self.analytical_radial(r[ref_idx]) < 0:
                R = -R

        return R

    # Rest of the class methods remain the same
    def angular_part(self, theta, phi):
        return sph_harm(self.m, self.l, phi, theta)

    def full_wavefunction_cartesian(self, x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)

        mask = (r < 1e-10)
        r[mask] = 1e-10
        theta[mask] = 0
        phi[mask] = 0

        R = self.analytical_radial(r)
        Y = self.angular_part(theta, phi)
        return R * Y

    def save_to_csv(self, r, theta, phi, base_path="Data Files/atomic orbitals"):
        # Create directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)

        # Calculate various components
        R_analytical = self.analytical_radial(r)
        R_numerical = self.numerical_radial(r)

        # Create dataframe for radial components
        radial_df = pd.DataFrame({
            'r': r,
            'R_analytical': R_analytical,
            'R_numerical': R_numerical,
            'P_analytical': R_analytical ** 2,
            'P_numerical': R_numerical ** 2,
            'Error': np.abs(R_analytical - R_numerical)
        })

        # Create grid for xy-plane
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        # Calculate full wavefunction in xy-plane
        psi_xy = self.full_wavefunction_cartesian(X, Y, Z)
        prob_density_xy = np.abs(psi_xy) ** 2

        # Create dataframe for xy-plane values
        xy_data = []
        for i in range(len(x)):
            for j in range(len(y)):
                xy_data.append({
                    'x': X[i, j],
                    'y': Y[i, j],
                    'psi_real': psi_xy[i, j].real,
                    'psi_imag': psi_xy[i, j].imag,
                    'probability_density': prob_density_xy[i, j]
                })
        xy_df = pd.DataFrame(xy_data)

        # Generate filenames
        base_filename = f"orbital_n{self.n}_l{self.l}_m{self.m}"
        radial_filename = os.path.join(base_path, f"{base_filename}_radial.csv")
        xy_filename = os.path.join(base_path, f"{base_filename}_xy_plane.csv")

        # Save to CSV files with headers
        with open(radial_filename, 'w') as f:
            f.write(f"# Hydrogen Atom Radial Wavefunction\n")
            f.write(f"# Quantum numbers: n={self.n}, l={self.l}, m={self.m}\n")
            f.write(f"# Energy (atomic units): {self.E}\n")
            f.write(f"# Columns: r, R_analytical, R_numerical, P_analytical, P_numerical, Error\n")
        radial_df.to_csv(radial_filename, mode='a', index=False)

        with open(xy_filename, 'w') as f:
            f.write(f"# Hydrogen Atom Wavefunction xy-plane slice\n")
            f.write(f"# Quantum numbers: n={self.n}, l={self.l}, m={self.m}\n")
            f.write(f"# Energy (atomic units): {self.E}\n")
            f.write(f"# Columns: x, y, psi_real, psi_imag, probability_density\n")
        xy_df.to_csv(xy_filename, mode='a', index=False)

        return radial_filename, xy_filename

    def get_rotated_plane_coordinates(self, size=10, points=200, theta_rot=0, phi_rot=0):
        """
        Generate coordinates for a rotated plane in 3D space.

        Args:
            size: Half-width of the plane (in Bohr radii)
            points: Number of points along each axis
            theta_rot: Polar angle of rotation (0 to pi)
            phi_rot: Azimuthal angle of rotation (0 to 2pi)
        """
        # Create a grid of points in the xy-plane
        x = np.linspace(-size, size, points)
        y = np.linspace(-size, size, points)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        # Create meshgrid of coordinates
        coords = np.stack([X, Y, Z], axis=-1)

        # Create rotation matrices
        # First rotate around y-axis (phi)
        Ry = np.array([
            [np.cos(phi_rot), 0, np.sin(phi_rot)],
            [0, 1, 0],
            [-np.sin(phi_rot), 0, np.cos(phi_rot)]
        ])

        # Then rotate around x-axis (theta)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_rot), -np.sin(theta_rot)],
            [0, np.sin(theta_rot), np.cos(theta_rot)]
        ])

        # Combined rotation
        R = Rx @ Ry

        # Apply rotation to all points
        rotated_coords = np.einsum('ij,mnj->mni', R, coords)

        return rotated_coords[..., 0], rotated_coords[..., 1], rotated_coords[..., 2]



def save_hydrogen_animation_analytical(n, l, m, plane_size=10, fps=30, duration_per_rotation=5, filename='hydrogen_orbital_rotation.gif'):
    """
    Create and save a smooth animation of the hydrogen orbital plane rotation
    First rotating around phi, then around theta

    Args:
        n, l, m: Quantum numbers
        plane_size: Half-width of the plane in Bohr radii
        fps: Frames per second
        duration_per_rotation: Duration of each rotation in seconds
        filename: Output filename
    """
    hw = HydrogenWavefunction(n, l, m)

    # Create figure for animation
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Set up the 3D plot with adjusted limits based on plane_size
    lim = plane_size * 1.2  # Add 20% margin
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('x (Bohr radii)')
    ax.set_ylabel('y (Bohr radii)')
    ax.set_zlabel('z (Bohr radii)')
    ax.set_box_aspect([1, 1, 1])

    # Add coordinate axes scaled to plane size
    origin = np.zeros(3)
    axis_length = plane_size * 1.2
    ax.quiver(origin[0], origin[1], origin[2],
              axis_length, 0, 0, color='r', alpha=0.5, arrow_length_ratio=0.1)
    ax.quiver(origin[0], origin[1], origin[2],
              0, axis_length, 0, color='g', alpha=0.5, arrow_length_ratio=0.1)
    ax.quiver(origin[0], origin[1], origin[2],
              0, 0, axis_length, color='b', alpha=0.5, arrow_length_ratio=0.1)

    # Rest of the animation setup remains the same, but use plane_size parameter
    n_frames_per_rotation = int(fps * duration_per_rotation)
    total_frames = 2 * n_frames_per_rotation
    points = 50

    print("Pre-calculating frames...")
    frames_data = []

    # First rotation: around phi with fixed theta = π/2
    phi_values = np.linspace(0, 2 * np.pi, n_frames_per_rotation)
    theta_fixed = np.pi / 2

    for i, phi in enumerate(phi_values):
        if i % 10 == 0:
            print(f"Calculating phi rotation frame {i + 1}/{n_frames_per_rotation}")

        X, Y, Z = hw.get_rotated_plane_coordinates(size=plane_size, points=points,
                                                   theta_rot=theta_fixed,
                                                   phi_rot=phi)
        psi_plane = hw.full_wavefunction_cartesian(X, Y, Z)
        prob_plane = np.abs(psi_plane) ** 2
        prob_normalized = prob_plane / prob_plane.max()
        frames_data.append((X, Y, Z, prob_normalized, theta_fixed, phi))
        gc.collect()

    # Second rotation: around theta with fixed phi = 0
    theta_values = np.linspace(np.pi / 2, -np.pi / 2, n_frames_per_rotation)
    phi_fixed = 0

    for i, theta in enumerate(theta_values):
        if i % 10 == 0:
            print(f"Calculating theta rotation frame {i + 1}/{n_frames_per_rotation}")

        X, Y, Z = hw.get_rotated_plane_coordinates(size=plane_size, points=points,
                                                   theta_rot=theta,
                                                   phi_rot=phi_fixed)
        psi_plane = hw.full_wavefunction_cartesian(X, Y, Z)
        prob_plane = np.abs(psi_plane) ** 2
        prob_normalized = prob_plane / prob_plane.max()
        frames_data.append((X, Y, Z, prob_normalized, theta, phi_fixed))
        gc.collect()

    # Second rotation: around theta with fixed phi = 0
    theta_values = np.linspace(np.pi / 2, -np.pi / 2, n_frames_per_rotation)
    phi_fixed = 0

    for i, theta in enumerate(theta_values):
        if i % 10 == 0:
            print(f"Calculating theta rotation frame {i + 1}/{n_frames_per_rotation}")

        X, Y, Z = hw.get_rotated_plane_coordinates(size=10, points=points,
                                                   theta_rot=theta,
                                                   phi_rot=phi_fixed)
        psi_plane = hw.full_wavefunction_cartesian(X, Y, Z)
        prob_plane = np.abs(psi_plane) ** 2
        prob_normalized = prob_plane / prob_plane.max()
        frames_data.append((X, Y, Z, prob_normalized, theta, phi_fixed))
        gc.collect()

    print("Frame calculation complete!")

    # Initialize the surface plot
    surf = [ax.plot_surface(frames_data[0][0], frames_data[0][1], frames_data[0][2],
                            facecolors=plt.cm.viridis(frames_data[0][3]),
                            alpha=0.8)]

    # Add colorbar
    m_bar = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    m_bar.set_array([0, 1])
    plt.colorbar(m_bar, ax=ax, label='Normalized Probability Density')

    def update(frame):
        # Clear previous surface
        surf[0].remove()

        # Get frame data
        X, Y, Z, prob_normalized, theta, phi = frames_data[frame]

        # Plot new surface
        surf[0] = ax.plot_surface(X, Y, Z,
                                  facecolors=plt.cm.viridis(prob_normalized),
                                  alpha=0.8)

        # Update title with current rotation information
        if frame < n_frames_per_rotation:
            rotation_type = "φ rotation"
        else:
            rotation_type = "θ rotation"

        ax.set_title(f'Probability Density of ({n}, {l}, {m}) H orbital in Rotated Plane\n' +
                     f'θ={theta:.2f}, φ={phi:.2f}\n{rotation_type}')
        return surf

    # Create and save animation
    print(f"Saving animation to {filename}...")
    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps)

    try:
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer,
                  progress_callback=lambda i, n: print(f"Saving frame {i + 1}/{n}"))
        print("Animation saved successfully!")
    except Exception as e:
        print(f"Error saving animation: {e}")
    finally:
        plt.close(fig)
        del frames_data
        del anim
        gc.collect()


def vectorized_numerical_radial(self, r):
    """
    Vectorized version of numerical_radial that handles array inputs.
    Ensures proper sorting of r values for solve_ivp.
    """
    # Handle scalar input
    if np.isscalar(r):
        r = np.array([r])
        scalar_input = True
    else:
        r = np.asarray(r)
        scalar_input = False

    # Remove any duplicate values and sort
    r_unique = np.unique(r)

    # Create mapping from original values to unique sorted values
    mapping = {val: i for i, val in enumerate(r_unique)}
    inverse_mapping = np.array([mapping[val] for val in r.flatten()])

    def radial_equation(r, y):
        if r < 1e-10:
            r = 1e-10
        V = -1.0 / r
        L_term = self.l * (self.l + 1) / (r * r)
        return [y[1], (2.0 * (V - self.E) + L_term) * y[0]]

    # Initial conditions near r=0
    r0 = 1e-5
    u0 = r0 ** (self.l + 1)
    du0 = (self.l + 1) * r0 ** self.l

    # Add r0 to evaluation points if needed
    if r_unique[0] > r0:
        r_eval = np.concatenate(([r0], r_unique))
    else:
        r_eval = r_unique

    # Solve up to the maximum r value
    solution = solve_ivp(
        radial_equation,
        t_span=(r0, r_eval[-1]),
        y0=[u0, du0],
        t_eval=r_eval,
        method='RK45',
        rtol=1e-12,
        atol=1e-12
    )

    # Convert from u(r) to R(r)
    R = np.where(r_eval > 1e-10, solution.y[0] / r_eval, 0)

    # Calculate normalization integral
    norm_integral = integrate.simps(R * R * r_eval * r_eval, r_eval)

    # Normalize the wavefunction
    R = R / np.sqrt(norm_integral)

    # Ensure consistent sign with analytical solution
    ref_point = self.n * 0.5
    ref_idx = np.abs(r_eval - ref_point).argmin()
    if ref_idx > 0:
        if R[ref_idx] * self.analytical_radial(r_eval[ref_idx]) < 0:
            R = -R

    # Remove the r0 point if we added it
    if r_eval[0] == r0 and r_unique[0] > r0:
        R = R[1:]

    # Map back to original r values
    R_mapped = R[inverse_mapping].reshape(r.shape)

    # Return scalar if input was scalar
    if scalar_input:
        return R_mapped[0]
    return R_mapped


def save_hydrogen_animation_compared(n, l, m, plane_size=10, fps=30, duration_per_rotation=5,
                                     filename='hydrogen_orbital_rotation_numerical.gif'):
    """
    Create and save a smooth animation of the hydrogen orbital plane rotation
    using the numerical solution. Shows both analytical and numerical solutions
    side by side for comparison.
    """
    hw = HydrogenWavefunction(n, l, m)

    # Replace the numerical_radial method with the vectorized version
    hw.numerical_radial = vectorized_numerical_radial.__get__(hw)

    # Create figure for animation with two subplots
    fig = plt.figure(figsize=(20, 10), dpi=100)

    # Create a gridspec with space for the legend at the top
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 10])

    # Create legend axis spanning both columns at top
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis('off')

    # Create dummy scatter plots for legend
    ax_legend.scatter([], [], c='blue', label='Analytical Solution')
    ax_legend.scatter([], [], c='red', label='Numerical Solution')

    # Create the main subplot axes
    ax_analytical = fig.add_subplot(gs[1, 0], projection='3d')
    ax_numerical = fig.add_subplot(gs[1, 1], projection='3d')

    # Add subtitles to each plot
    ax_analytical.set_title('Analytical Solution', y=1.1, pad=10, fontsize=12)
    ax_numerical.set_title('Numerical Solution', y=1.1, pad=10, fontsize=12)

    # Update plot limits based on plane_size
    lim = plane_size * 1.2  # Add 20% margin
    for ax in [ax_analytical, ax_numerical]:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

        # Scale coordinate axes
        origin = np.zeros(3)
        axis_length = plane_size * 1.2
        ax.quiver(origin[0], origin[1], origin[2],
                  axis_length, 0, 0, color='r', alpha=0.5, arrow_length_ratio=0.1)
        ax.quiver(origin[0], origin[1], origin[2],
                  0, axis_length, 0, color='g', alpha=0.5, arrow_length_ratio=0.1)
        ax.quiver(origin[0], origin[1], origin[2],
                  0, 0, axis_length, color='b', alpha=0.5, arrow_length_ratio=0.1)

    # Calculate frame parameters
    n_frames_per_rotation = int(fps * duration_per_rotation)
    total_frames = 2 * n_frames_per_rotation  # Total frames for both rotations
    points = 50  # Number of points in the plane grid

    # Pre-calculate frames
    print("Pre-calculating frames...")
    frames_data = []

    # First rotation: around phi with fixed theta = π/2
    phi_values = np.linspace(0, 2 * np.pi, n_frames_per_rotation)
    theta_fixed = np.pi / 2

    for i, phi in enumerate(phi_values):
        if i % 10 == 0:
            print(f"Calculating phi rotation frame {i + 1}/{n_frames_per_rotation}")

        X, Y, Z = hw.get_rotated_plane_coordinates(size=10, points=points,
                                                   theta_rot=theta_fixed,
                                                   phi_rot=phi)

        # Calculate both analytical and numerical solutions
        psi_analytical = hw.full_wavefunction_cartesian(X, Y, Z)

        r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        theta = np.arccos(Z / np.where(r < 1e-10, 1e-10, r))
        phi_coords = np.arctan2(Y, X)

        R_numerical = hw.numerical_radial(r.flatten()).reshape(r.shape)
        Y_angular = hw.angular_part(theta, phi_coords)
        psi_numerical = R_numerical * Y_angular

        # Calculate probability densities
        prob_analytical = np.abs(psi_analytical) ** 2
        prob_numerical = np.abs(psi_numerical) ** 2

        # Normalize probabilities
        prob_analytical_norm = prob_analytical / prob_analytical.max()
        prob_numerical_norm = prob_numerical / prob_numerical.max()

        frames_data.append((X, Y, Z, prob_analytical_norm, prob_numerical_norm, theta_fixed, phi))
        gc.collect()

    # Second rotation: around theta with fixed phi = 0
    theta_values = np.linspace(np.pi / 2, -np.pi / 2, n_frames_per_rotation)
    phi_fixed = 0

    for i, theta in enumerate(theta_values):
        if i % 10 == 0:
            print(f"Calculating theta rotation frame {i + 1}/{n_frames_per_rotation}")

        X, Y, Z = hw.get_rotated_plane_coordinates(size=10, points=points,
                                                   theta_rot=theta,
                                                   phi_rot=phi_fixed)

        # Calculate both analytical and numerical solutions
        psi_analytical = hw.full_wavefunction_cartesian(X, Y, Z)

        r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        theta_coords = np.arccos(Z / np.where(r < 1e-10, 1e-10, r))
        phi_coords = np.arctan2(Y, X)

        R_numerical = hw.numerical_radial(r.flatten()).reshape(r.shape)
        Y_angular = hw.angular_part(theta_coords, phi_coords)
        psi_numerical = R_numerical * Y_angular

        # Calculate probability densities
        prob_analytical = np.abs(psi_analytical) ** 2
        prob_numerical = np.abs(psi_numerical) ** 2

        # Normalize probabilities
        prob_analytical_norm = prob_analytical / prob_analytical.max()
        prob_numerical_norm = prob_numerical / prob_numerical.max()

        frames_data.append((X, Y, Z, prob_analytical_norm, prob_numerical_norm, theta, phi_fixed))
        gc.collect()

    print("Frame calculation complete!")

    # Initialize the surface plots
    surfs = [
        ax_analytical.plot_surface(frames_data[0][0], frames_data[0][1], frames_data[0][2],
                                   facecolors=plt.cm.viridis(frames_data[0][3]),
                                   alpha=0.8),
        ax_numerical.plot_surface(frames_data[0][0], frames_data[0][1], frames_data[0][2],
                                  facecolors=plt.cm.viridis(frames_data[0][4]),
                                  alpha=0.8)
    ]

    # Add colorbar on the right
    m_bar = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    m_bar.set_array([0, 1])
    cbar = plt.colorbar(m_bar, ax=[ax_analytical, ax_numerical],
                        label='Normalized Probability Density')

    def update(frame):
        # Clear previous surfaces
        surfs[0].remove()
        surfs[1].remove()

        # Get frame data
        X, Y, Z, prob_analytical, prob_numerical, theta, phi = frames_data[frame]

        # Plot new surfaces
        surfs[0] = ax_analytical.plot_surface(X, Y, Z,
                                              facecolors=plt.cm.viridis(prob_analytical),
                                              alpha=0.8)
        surfs[1] = ax_numerical.plot_surface(X, Y, Z,
                                             facecolors=plt.cm.viridis(prob_numerical),
                                             alpha=0.8)

        # Update main title with current rotation information
        if frame < n_frames_per_rotation:
            rotation_type = "φ rotation"
        else:
            rotation_type = "θ rotation"

        fig.suptitle(f'Probability Density of ({n}, {l}, {m}) H Orbital\nθ={theta:.2f}, φ={phi:.2f}\n{rotation_type}',
                     fontsize=14, y=0.95)

        return surfs

    # Create and save animation
    print(f"Saving animation to {filename}...")
    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps)

    try:
        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer,
                  progress_callback=lambda i, n: print(f"Saving frame {i + 1}/{n}"))
        print("Animation saved successfully!")
    except Exception as e:
        print(f"Error saving animation: {e}")
    finally:
        plt.close(fig)
        del frames_data
        del anim
        gc.collect()


def create_hydrogen_visualization(n, l, m, plane_size=10, save_animation=False):
    """
    Create interactive visualization with optional animation saving

    Args:
        n, l, m: Quantum numbers
        plane_size: Half-width of the plane in Bohr radii
        save_animation: Whether to save animation
    """
    if save_animation:
        save_hydrogen_animation_analytical(n, l, m, plane_size=plane_size)

    # Switch back to interactive backend if needed
    if matplotlib.get_backend() != 'TkAgg':
        matplotlib.use('TkAgg')

    hw = HydrogenWavefunction(n, l, m)

    # Create two separate figures
    fig_radial = plt.figure(figsize=(12, 6))
    fig_3d = plt.figure(figsize=(10, 10))

    # Setup radial functions figure
    gs_radial = fig_radial.add_gridspec(1, 2)
    ax_radial = fig_radial.add_subplot(gs_radial[0, 0])
    ax_prob = fig_radial.add_subplot(gs_radial[0, 1])

    # Setup 3D figure
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # Calculate radial functions
    r = np.linspace(0.01, 15, 500)
    R_analytical = hw.analytical_radial(r)
    R_numerical = hw.numerical_radial(r)

    # Calculate normalization integrals
    r_integrate = np.linspace(0.01, 100, 1000)
    R_analytical_integrate = hw.analytical_radial(r_integrate)
    R_numerical_integrate = hw.numerical_radial(r_integrate)

    analytical_integral = integrate.trapz(R_analytical_integrate * R_analytical_integrate * r_integrate * r_integrate,
                                          r_integrate)
    numerical_integral = integrate.trapz(R_numerical_integrate * R_numerical_integrate * r_integrate * r_integrate,
                                         r_integrate)

    print(f'Analytical integral = {analytical_integral:.6f}')
    print(f'Numerical integral = {numerical_integral:.6f}')

    # Plot radial wavefunctions
    ax_radial.plot(r, R_analytical, 'b-', label='Analytical')
    ax_radial.plot(r, R_numerical, 'r--', label='Numerical')
    ax_radial.set_xlabel('r (Bohr radii)')
    ax_radial.set_ylabel('R(r)')
    ax_radial.set_title(f'Radial Wavefunction (n={n}, l={l}, m={m})')
    ax_radial.legend()
    ax_radial.grid(True)

    # Plot probability densities
    ax_prob.plot(r, R_analytical ** 2, 'b-', label='Analytical')
    ax_prob.plot(r, R_numerical ** 2, 'r--', label='Numerical')
    ax_prob.set_xlabel('r (Bohr radii)')
    ax_prob.set_ylabel('|R(r)|²')
    ax_prob.set_title('Radial Probability Density')
    ax_prob.legend()
    ax_prob.grid(True)

    # Adjust radial figure layout
    fig_radial.tight_layout()

    # Initial rotation angles
    theta_rot_init = np.pi / 2  # Start at π/2
    phi_rot_init = 0


    # Update 3D plot limits based on plane_size
    lim = plane_size * 1.2
    ax_3d.set_xlim(-lim, lim)
    ax_3d.set_ylim(-lim, lim)
    ax_3d.set_zlim(-lim, lim)

    # Scale coordinate axes
    origin = np.zeros(3)
    axis_length = plane_size * 1.2
    ax_3d.quiver(origin[0], origin[1], origin[2],
                 axis_length, 0, 0, color='r', alpha=0.5, arrow_length_ratio=0.1)
    ax_3d.quiver(origin[0], origin[1], origin[2],
                 0, axis_length, 0, color='g', alpha=0.5, arrow_length_ratio=0.1)
    ax_3d.quiver(origin[0], origin[1], origin[2],
                 0, 0, axis_length, color='b', alpha=0.5, arrow_length_ratio=0.1)

    # Use plane_size in get_rotated_plane_coordinates calls
    X, Y, Z = hw.get_rotated_plane_coordinates(size=plane_size, points=100,
                                               theta_rot=theta_rot_init,
                                               phi_rot=phi_rot_init)

    # Calculate initial probability density for the plane
    psi_plane = hw.full_wavefunction_cartesian(X, Y, Z)
    prob_plane = np.abs(psi_plane) ** 2

    # Create initial 3D plot with just the plane
    plane = ax_3d.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(prob_plane / prob_plane.max()),
                               alpha=0.8)

    # Add colorbar
    m_bar = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    m_bar.set_array(prob_plane / prob_plane.max())
    plt.colorbar(m_bar, ax=ax_3d, label='Normalized Probability Density')

    # Adjust 3D figure to make room for sliders
    fig_3d.subplots_adjust(bottom=0.25)

    # Create slider axes
    ax_theta = fig_3d.add_axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgoldenrodyellow')
    ax_phi = fig_3d.add_axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgoldenrodyellow')

    # Create sliders
    s_theta = Slider(
        ax_theta,
        'θ (theta)',
        0,
        np.pi,
        valinit=theta_rot_init,
        color='royalblue'
    )

    s_phi = Slider(
        ax_phi,
        'φ (phi)',
        0,
        2 * np.pi,
        valinit=phi_rot_init,
        color='royalblue'
    )

    def update(val):
        # Get current slider values
        theta_rot = s_theta.val
        phi_rot = s_phi.val

        # Clear the current plane
        for coll in ax_3d.collections:
            if isinstance(coll, plt.cm.ScalarMappable):
                coll.remove()

        # Calculate new plane coordinates and probability density
        X, Y, Z = hw.get_rotated_plane_coordinates(size=10, points=100,
                                                   theta_rot=theta_rot,
                                                   phi_rot=phi_rot)
        psi_plane = hw.full_wavefunction_cartesian(X, Y, Z)
        prob_plane = np.abs(psi_plane) ** 2

        # Plot new plane
        plane = ax_3d.plot_surface(X, Y, Z,
                                   facecolors=plt.cm.viridis(prob_plane / prob_plane.max()),
                                   alpha=0.8)

        # Update title
        ax_3d.set_title(f'Probability Density in Rotated Plane\n' +
                        f'θ={theta_rot:.2f}, φ={phi_rot:.2f}')

        fig_3d.canvas.draw_idle()

    # Register the update function with both sliders
    s_theta.on_changed(update)
    s_phi.on_changed(update)

    plt.show()


if __name__ == "__main__":
    # Example usage
    n, l, m = 5, 3, 2

    # For interactive visualization only:
    # create_hydrogen_visualization(n, l, m, save_animation=False)

    # For saving animation:
    # create_hydrogen_visualization(n, l, m, save_animation=True)

    # Or to just save the animation without visualization:
    save_hydrogen_animation_compared(n, l, m, fps=30, duration_per_rotation=5)