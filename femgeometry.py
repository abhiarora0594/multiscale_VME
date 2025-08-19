import numpy as np 
import math
from mesh import Mesh2D
from shape_function import ShapeFunction2D

# FEM and Geometry mixed class
class FEMGeometry:

    def __init__(self, mesh: Mesh2D, dof_per_node = 2, ngp = 4):

        # geometry
        self.mesh = mesh

        # shape functions and their local derivatives
        self.shape_fn = ShapeFunction2D(mesh.element_type)

        # 1st dof in x direction and 2nd dof in y direction
        self.dof_per_node = dof_per_node

        # generate dof connectivity coarse
        self.dof_conn_coarse = self._generate_dof_connectivity_coarse()

        # generate dof connectivity fine
        self.dof_conn_fine = self._generate_dof_connectivity_fine()

        # generate dof boundary connectivity coarse
        self.dof_bound_conn_coarse = self._generate_dof_boundary_connectivity_coarse()

        # generate dof boundary connectivity fine
        self.dof_bound_conn_fine = self._generate_dof_boundary_connectivity_fine()

        # no of gps for integration
        self.ngp = ngp

        # gp and wt list for Gauss-Quadrature
        self.gp_list , self.wt_list = self.gauss_points()
    
    def _generate_dof_connectivity_coarse(self):

        m,n = self.mesh.elements.shape
        
        dof_conn = np.zeros((m,self.dof_per_node*n))

        for i in range(m):
            for j in range(n):
                for k in range(self.dof_per_node):
                    dof_conn[i][self.dof_per_node*j+k] = self.dof_per_node*self.mesh.elements[i][j]+ k
        
        return np.array(dof_conn,dtype = np.int64)
    
    def _generate_dof_connectivity_fine(self):

        m,n = self.mesh.elements_m.shape
        
        dof_conn = np.zeros((m,self.dof_per_node*n))

        for i in range(m):
            for j in range(n):
                for k in range(self.dof_per_node):
                    dof_conn[i][self.dof_per_node*j+k] = self.dof_per_node*self.mesh.elements_m[i][j]+ k
        
        return np.array(dof_conn,dtype = np.int64)
    
    def _generate_dof_boundary_connectivity_coarse(self):

        m,n = self.mesh.boundary_elements.shape

        dof_bound_conn_coarse = np.zeros((m,self.dof_per_node*n))

        for i in range(m):
            for j in range(n):
                for k in range(self.dof_per_node):
                    dof_bound_conn_coarse[i][self.dof_per_node*j+k] = self.dof_per_node*self.mesh.boundary_elements[i][j]+ k

        return np.array(dof_bound_conn_coarse, dtype = np.int64)
    
    def _generate_dof_boundary_connectivity_fine(self):

        m,n = self.mesh.boundary_elements.shape

        dof_bound_conn_fine = np.zeros((m,self.dof_per_node*n))

        for i in range(m):
            for j in range(n):
                for k in range(self.dof_per_node):
                    dof_bound_conn_fine[i][self.dof_per_node*j+k] = self.dof_per_node*self.mesh.boundary_elements_m[i][j]+ k

        return np.array(dof_bound_conn_fine, dtype = np.int64)
    
    def gauss_points(self):

        gp_list = []
        wt_list = []

        if self.ngp == 4:
            gp_list.append([-1/math.sqrt(3), -1/math.sqrt(3)])
            gp_list.append([1/math.sqrt(3), -1/math.sqrt(3)])
            gp_list.append([1/math.sqrt(3), 1/math.sqrt(3)])
            gp_list.append([-1/math.sqrt(3), 1/math.sqrt(3)])

            for i in range(self.ngp):
                wt_list.append(1.0)
        else:
            raise ValueError("Only implemented for 4 gauss points")

        return np.array(gp_list), np.array(wt_list)


    def get_global_shape_gradients(self, coords, xi, eta):
        """Compute ∇N and |J| at (xi, eta) in element eid."""

        # Compute ∂N/∂ξ, ∂N/∂η
        dN_dxi_eta = self.shape_fn.gradients(xi, eta)  # shape (nen, 2)

        # Compute Jacobian J = dX/dξ
        J = coords.T @ dN_dxi_eta  # (2xN) @ (N x 2) → (2x2)

        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError(f"Invalid element : det(J) ≤ 0")

        J_inv = np.linalg.inv(J)

        # Compute (∇N).T =  (∂N/∂ξ).T @ J⁻¹
        gradN_global = dN_dxi_eta @ J_inv  # (N x 2)

        return gradN_global, detJ
    
    def inverse_coarse_mapping(self, X_g, X_c_e):

        # only four corner nodes are required
        X_c = X_c_e[0:4,:]

        # coefficients
        a_0 = 0.25*(X_c[0][0] + X_c[1][0] + X_c[2][0] + X_c[3][0])
        b_0 = 0.25*(X_c[0][1] + X_c[1][1] + X_c[2][1] + X_c[3][1])

        a_1 = 0.25*(-X_c[0][0] + X_c[1][0] + X_c[2][0] - X_c[3][0])
        b_1 = 0.25*(-X_c[0][1] + X_c[1][1] + X_c[2][1] - X_c[3][1])

        a_2 = 0.25*(X_c[0][0] - X_c[1][0] + X_c[2][0] - X_c[3][0])
        b_2 = 0.25*(X_c[0][1] - X_c[1][1] + X_c[2][1] - X_c[3][1])

        a_3 = 0.25*(-X_c[0][0] - X_c[1][0] + X_c[2][0] + X_c[3][0])
        b_3 = 0.25*(-X_c[0][1] - X_c[1][1] + X_c[2][1] + X_c[3][1])

        j0 = a_1*b_3 - a_3*b_1
        transformation_mat = np.array([[b_3, -a_3],[-b_1, a_1]])
        
        xi_eta_bar = np.multiply(1/j0, np.matmul(transformation_mat,X_g))
        alpha = (a_1*b_2 - a_2*b_1)/j0
        gamma = (a_2*b_3 - a_3*b_2)/j0

        # geodesic parameters
        xi_geo = xi_eta_bar[0] - gamma*xi_eta_bar[0]*xi_eta_bar[1] + (alpha*gamma*xi_eta_bar[0]**2 * xi_eta_bar[1] + gamma**2*xi_eta_bar[0]*xi_eta_bar[1]**2) \
                 - (alpha**2 * gamma*xi_eta_bar[0]**3 * xi_eta_bar[1] + 2.0*alpha*gamma**2 * xi_eta_bar[0]**2 * xi_eta_bar[1]**2 + gamma**3 *xi_eta_bar[0]*xi_eta_bar[1]**3)
        
        eta_geo = xi_eta_bar[1] - alpha*xi_eta_bar[0]*xi_eta_bar[1] + (alpha**2*xi_eta_bar[0]**2 * xi_eta_bar[1] + alpha*gamma*xi_eta_bar[0]*xi_eta_bar[1]**2) \
                 - (alpha**3*xi_eta_bar[0]**3 * xi_eta_bar[1] + 2.0*alpha**2 *gamma * xi_eta_bar[0]**2 * xi_eta_bar[1]**2 + alpha*gamma**2 *xi_eta_bar[0]*xi_eta_bar[1]**3)
        
        # hat coefficients
        a_hat_0 = 0 #0.25*((-1) + 1 + 1 + (-1))
        b_hat_0 = 0 #0.25*((-1) + (-1) + 1 +1)

        a_hat_1 = 1 #0.25*(-(-1) + 1 + 1 - (-1))
        b_hat_1 = 0 #0.25*(-(-1) + (-1) + 1 -1)

        a_hat_2 = 0 #0.25*((-1) - 1 + 1 - (-1))
        b_hat_2 = 0 #0.25*((-1) - (-1) + 1 - 1)

        a_hat_3 = 0 #0.25*(-(-1) - 1 + 1 + (-1))
        b_hat_3 = 1 #0.25*(-(-1) - (-1) + 1 + 1)

        xi_g = a_hat_0 + a_hat_1*xi_geo + a_hat_2*xi_geo*eta_geo + a_hat_3*eta_geo
        eta_g = b_hat_0 + b_hat_1*xi_geo + b_hat_2*xi_geo*eta_geo + b_hat_3*eta_geo
        # it seems that no correction happens and xi_g = xi_geo and eta_g = eta_geo with above hat coefficients

        return xi_g, eta_g
