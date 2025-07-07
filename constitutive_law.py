import numpy as np
from mesh import Mesh2D

# Constitutive Law class    
class Constitutive_Law:

    def __init__(self, mesh: Mesh2D, MATERIAL_MODEL):
        self.material_model = MATERIAL_MODEL
        self.mesh = mesh
    
    def get_first_piola_stress(self,F_def,lambda_,mu_):

        F_def = F_def.reshape(2,2)

        det_F = np.linalg.det(F_def)

        P_def = np.multiply(lambda_*det_F - mu_, np.transpose(np.inv(F_def))) + np.multiply(mu_,F_def)

        return P_def.reshape(-1,1)
    
    def assign_neohookean_properties(self,mu_,lambda_,rho_):

        self.lambda_list = np.zeros((self.mesh.elements_m.shape[0],1))
        self.mu_list = np.zeros((self.mesh.elements_m.shape[0],1))
        self.rho_list = np.zeros((self.mesh.elements_m.shape[0],1))

        for ie in range(self.mesh.elements_m.shape[0]):
            self.lambda_list[ie] = lambda_
            self.mu_list[ie] = mu_
            self.rho_list[ie] = rho_
        