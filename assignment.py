import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simpson  

class MobiusStrip:
    def __init__(self, R=1.0, w=0.2, n=100):
        """
        Initialize the Mobius strip parameters.
        R: Radius of the center circle
        w: Width of the strip
        n: Resolution (number of mesh points per dimension)
        """
        self.R = R
        self.w = w
        self.n = n
        self.u, self.v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(-w/2, w/2, n))
        self.x, self.y, self.z = self._compute_mesh()

    def _compute_mesh(self):
        """
        Compute 3D coordinates using the parametric equations.
        """
        u, v = self.u, self.v
        half_u = u / 2.0
        x = (self.R + v * np.cos(half_u)) * np.cos(u)
        y = (self.R + v * np.cos(half_u)) * np.sin(u)
        z = v * np.sin(half_u)
        return x, y, z

    def compute_surface_area(self):
        """
        Approximate the surface area using numerical integration.
        Use the norm of the cross product of partial derivatives.
        """
        du = 2 * np.pi / (self.n - 1)
        dv = self.w / (self.n - 1)

        # Compute partial derivatives
        x_u = np.gradient(self.x, du, axis=1)
        x_v = np.gradient(self.x, dv, axis=0)
        y_u = np.gradient(self.y, du, axis=1)
        y_v = np.gradient(self.y, dv, axis=0)
        z_u = np.gradient(self.z, du, axis=1)
        z_v = np.gradient(self.z, dv, axis=0)

        # Cross product of partial derivatives
        cross_prod = np.cross(np.stack((x_u, y_u, z_u), axis=-1),
                              np.stack((x_v, y_v, z_v), axis=-1))

        # Norm of cross product = differential area
        area_density = np.linalg.norm(cross_prod, axis=-1)

        area = simpson(simpson(area_density, self.v[:, 0]), self.u[0])
        return area

    def compute_edge_length(self):
        """
        Approximate the total edge length by tracing both boundaries.
        """
        # Top and bottom edges (v = ±w/2)
        edges = []
        for sign in [-1, 1]:
            v_edge = sign * self.w / 2
            u_vals = np.linspace(0, 2 * np.pi, self.n)
            x = (self.R + v_edge * np.cos(u_vals / 2)) * np.cos(u_vals)
            y = (self.R + v_edge * np.cos(u_vals / 2)) * np.sin(u_vals)
            z = v_edge * np.sin(u_vals / 2)
            points = np.vstack((x, y, z)).T
            dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
            edges.append(np.sum(dists))
        return sum(edges)

    def plot(self):
        """
        Display a 3D plot of the Möbius strip.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x, self.y, self.z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title("Möbius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    mobius = MobiusStrip(R=1.0, w=0.3, n=200)
    area = mobius.compute_surface_area()
    edge_length = mobius.compute_edge_length()
    print(f"Surface Area ≈ {area:.4f}")
    print(f"Edge Length ≈ {edge_length:.4f}")
    mobius.plot()
