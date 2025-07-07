import numpy as np

# shape function class
class ShapeFunction2D:
    def __init__(self, element_type="Q1"):
        self.element_type = element_type.upper()
        if self.element_type not in ("Q1", "Q2"):
            raise ValueError("element_type must be 'Q1' or 'Q2'")

    def evaluate(self, xi, eta):
        if self.element_type == "Q1":
            return self._q1_shape_functions(xi, eta)
        elif self.element_type == "Q2":
            return self._q2_shape_functions(xi, eta)

    def gradients(self, xi, eta):
        if self.element_type == "Q1":
            return self._q1_gradients(xi, eta)
        elif self.element_type == "Q2":
            return self._q2_gradients(xi, eta)

    def _q1_shape_functions(self, xi, eta):
        N = np.array([
            0.25 * (1 - xi) * (1 - eta),  # node 0
            0.25 * (1 + xi) * (1 - eta),  # node 1
            0.25 * (1 + xi) * (1 + eta),  # node 2
            0.25 * (1 - xi) * (1 + eta)   # node 3
        ])
        return N

    def _q1_gradients(self, xi, eta):
        dN_dxi = np.array([
            [-0.25 * (1 - eta), -0.25 * (1 - xi)],  # node 0
            [ 0.25 * (1 - eta), -0.25 * (1 + xi)],  # node 1
            [ 0.25 * (1 + eta),  0.25 * (1 + xi)],  # node 2
            [-0.25 * (1 + eta),  0.25 * (1 - xi)]   # node 3
        ])
        return dN_dxi

    def _q2_shape_functions(self, xi, eta):
        N = np.zeros(9)
        N[0] = 0.25 * (xi - 1) * (eta - 1) * xi * eta # left - bottom
        N[1] = 0.25 * (xi + 1) * (eta - 1) * xi * eta # right - bottom
        N[2] = 0.25 * (xi + 1) * (eta + 1) * xi * eta # right - top
        N[3] = 0.25 * (xi - 1) * (eta + 1) * xi * eta # left - top
        N[4] = 0.5 * (1 - xi**2) * (eta - 1) * eta # mid - bottom
        N[5] = 0.5 * (xi + 1) * (1 - eta**2) * xi # right - mid
        N[6] = 0.5 * (1 - xi**2) * (eta + 1) * eta # mid - top
        N[7] = 0.5 * (xi - 1) * (1 - eta**2) * xi # left - mid
        N[8] = (1 - xi**2) * (1 - eta**2) # center
        
        return N

    def _q2_gradients(self, xi, eta):
        # Partial derivatives ∂N/∂xi and ∂N/∂eta (9 x 2 array)
        dN = np.zeros((9, 2))

        # Derivatives manually derived (could also use sympy)
        dN[0, 0] = 0.25 * (2 * xi - 1) * (eta - 1) * eta
        dN[0, 1] = 0.25 * (xi - 1) * (2 * eta - 1) * xi

        dN[1, 0] = 0.25 * (2 * xi + 1) * (eta - 1) * eta
        dN[1, 1] = 0.25 * (xi + 1) * (2 * eta - 1) * xi

        dN[2, 0] = 0.25 * (2 * xi + 1) * (eta + 1) * eta
        dN[2, 1] = 0.25 * (xi + 1) * (2 * eta + 1) * xi

        dN[3, 0] = 0.25 * (2 * xi - 1) * (eta + 1) * eta
        dN[3, 1] = 0.25 * (xi - 1) * (2 * eta + 1) * xi

        dN[4, 0] = -xi * (eta - 1) * eta
        dN[4, 1] = 0.5 * (1 - xi**2) * (2 * eta - 1)

        dN[5, 0] = 0.5 * (2 * xi + 1) * (1 - eta**2)
        dN[5, 1] = - (xi + 1) * xi * eta

        dN[6, 0] = -xi * (eta + 1) * eta
        dN[6, 1] = 0.5 * (1 - xi**2) * (2 * eta + 1)

        dN[7, 0] = 0.5 * (2 * xi - 1) * (1 - eta**2)
        dN[7, 1] = - (xi - 1) * xi * eta

        dN[8, 0] = -2 * xi * (1 - eta**2)
        dN[8, 1] = -2 * eta * (1 - xi**2)

        return dN