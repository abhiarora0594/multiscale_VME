from femgeometry import FEMGeometry
from constitutive_law import Constitutive_Law
from multiscale_solver import Data_Manager, CoarseScaleSolver, FineScaleSolver

class MultiScaleAlgorithm:
    
    def __init__(self, femgeometry: FEMGeometry, constitutivelaw: Constitutive_Law, CFL, P_,  T_TOTAL, TOL):
        
        self.femgeometry = femgeometry
        self.constitutivelaw = constitutivelaw
        self.CFL = CFL # CFL
        self.P_ = P_ # sub-step ratio
        self.T_TOTAl = T_TOTAL # total time
        self.TOL = TOL # tolerance for VME
        self.delta_t = 0.0 # time increment

        # get integration constants
        self.q0,self.q1,self.q2 = self.get_integration_scheme_constants()
    
    def get_integration_scheme_constants(self):

        q1 = (1-2*self.P_)/(2*self.P_*(1-self.P_))
        q2 = 0.5 - self.P_*q1
        q0 = -q1 - q2 + 0.5

        return q0,q1,q2