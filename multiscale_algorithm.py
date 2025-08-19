from femgeometry import FEMGeometry
from constitutive_law import Constitutive_Law
from multiscale_solver import Data_Manager, CoarseScaleSolver, FineScaleSolver
import numpy as np

class MultiScaleAlgorithm:
    
    def __init__(self, femgeometry: FEMGeometry, constitutivelaw: Constitutive_Law, CFL, P_,  T_TOTAL, TOL, NSTEPS):
        
        self.femgeometry = femgeometry
        self.constitutivelaw = constitutivelaw
        self.CFL = CFL # CFL
        self.P_ = P_ # sub-step ratio
        self.T_TOTAl = T_TOTAL # total time
        self.TOL = TOL # tolerance for VME
        self.delta_t = 0.0 # time increment
        self.NSTEPS = NSTEPS
        self.step_no = 0
        self.t_sim = 0.0

        # get integration constants
        self.q0,self.q1,self.q2 = self.get_integration_scheme_constants()

        # setup the classes
        self.data_manger = Data_Manager(self.femgeometry)
        self.coarse_scale_solver =  CoarseScaleSolver(self.femgeometry, self.data_manger, self.constitutivelaw)
        self.fine_scale_solver =  FineScaleSolver(self.femgeometry, self.data_manger, self.constitutivelaw)
    
    def explicit_explicit_algorithm(self):

        # setup initial conditions (coarse and fine scale)
        self.apply_initial_conditions_coarse_scale()
        self.apply_initial_conditions_fine_scale()

        # get initial accelerations by iteratively solving fine and coarse scale problems
        self.get_initial_accelerations_coarse_and_fine_scales()

        # include boundary conditions in displacement and velocity updates
        # write output files
        # get a stable time integration step
        # create a case for heterogeneous material input parameters

        # while loop
        while (self.step_no < self.NSTEPS and self.t_sim < self.T_TOTAl):

            ## get stable time increment

            # update time and step no
            self.t_sim += self.delta_t
            self.step_no +=1
            
            ## constants required for time integration
            a0,a1,a2,a3,a4,a5,a6,a7 = self.get_integration_scheme_variables()

            ### ----------- #####
            ## sub-step integraion
            #### ----------- #### 

            ###  displacement fields update ###
            # displacement update coarse-scale
            self.coarse_scale_solver.update_coarse_displacement_substep(a0,a1)

            # displacement update fine-scale
            self.fine_scale_solver.update_fine_displacement_substep(a0,a1)

            ###  acceleration guess coarse-scale ###
            # acceleration update
            self.data_manger.a_f_it = self.data_manger.a_f.copy()
            self.coarse_scale_solver.update_coarse_acceleration_substep()

            # storing the iterates before updates
            self.data_manger.a_c_sub = self.data_manger.a_c_it.copy()
            self.data_manger.a_f_sub = self.data_manger.a_f_it.copy()

            # operator-split iterations for sub-step
            max_error = 1.0

            while (max_error > self.TOL):
                
                # acceleration fine-scale update
                self.fine_scale_solver.update_fine_acceleration_substep()

                # acceleration coarse-scale update
                self.coarse_scale_solver.update_coarse_acceleration_substep()

                ## estimate error ##
                max_error = self.data_manger.error_estimate_substep()

                # update acceleration iterates
                self.data_manger.a_c_sub = self.data_manger.a_c_it.copy()
                self.data_manger.a_f_sub = self.data_manger.a_f_it.copy()
            

            ###  velocity fields update ###
            # velocity update coarse-scale
            self.coarse_scale_solver.update_coarse_velocity_substep(a2)

            # velocity update fine-scale
            self.fine_scale_solver.update_fine_velocity_substep(a2)

            ### ----------- #####
            ## full-step integraion
            #### ----------- ####

            ### displacement field updates ###
            # displacement update coarse-scale
            self.coarse_scale_solver.update_coarse_displacement_fullstep(a3,a4)

            # displacement update fine-scale
            self.fine_scale_solver.update_fine_displacement_fullstep(a3,a4)

            # acceleration field guess coarse scale
            self.data_manger.a_f_it = self.data_manger.a_f_sub.copy()
            self.coarse_scale_solver.update_coarse_acceleration_fullstep()

            # storing the iterates before updates
            self.data_manger.a_c_up = self.data_manger.a_c_it.copy()
            self.data_manger.a_f_up = self.data_manger.a_f_it.copy()

            # operator-split iterations for full-step
            max_error = 1.0
            
            while (max_error > self.TOL):
                
                # acceleration update coarse-scale
                self.fine_scale_solver.update_fine_acceleration_fullstep()

                # acceleration update fine-scale
                self.coarse_scale_solver.update_coarse_acceleration_fullstep()

                ## estimate error ##
                self.data_manger.error_estimate_fullstep()

                # update acceleration iterates
                self.data_manger.a_c_up = self.data_manger.a_c_it.copy()
                self.data_manger.a_f_up = self.data_manger.a_f_it.copy()
            
            ### velocity field updates ###
            # velocity update coarse-scale
            self.coarse_scale_solver.update_coarse_velocity_fullstep(a5,a6,a7)

            # velocity update fine-scale
            self.fine_scale_solver.update_fine_velocity_fullstep(a5,a6,a7)
    

    def get_integration_scheme_constants(self):

        # these variables do not depend on the delta t (time increment)
        q1 = (1-2*self.P_)/(2*self.P_*(1-self.P_))
        q2 = 0.5 - self.P_*q1
        q0 = -q1 - q2 + 0.5

        return q0,q1,q2
    
    def get_integration_scheme_variables(self):
       
        # these variables depend on the delta t (time increment)
        a0 = self.P*self.delta_t
        a1 = 0.5*a0*a0
        a2 = 0.5*a0
        a3 = (1-self.P)*self.delta_t
        a4 = 0.5*((1-self.P)*self.delta_t)**2
        a5 = self.q0*a3
        a6 = (0.5+self.q1)*a3
        a7 = self.q2*a3

        return a0,a1,a2,a3,a4,a5,a6,a7
    
    def apply_initial_conditions_coarse_scale(self):
        pass

    def apply_initial_conditions_fine_scale(self):
        # initial displacements and velocities are zero
        self.data_manger.u_f = np.zeros((self.femgeomtery.mesh.nodes_m.shape[0],self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes_m.shape[1]))
        self.data_manger.v_f = np.zeros((self.femgeomtery.mesh.nodes_m.shape[0],self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes_m.shape[1]))

    def get_initial_accelerations_coarse_and_fine_scales(self):

        ###  pseudo displacement fields update (not actual) ###
        # this just stores the initial displacements in u_c_sub and u_f_sub 
        # so that acceleration updates can access these fields

        # displacement update coarse-scale
        self.coarse_scale_solver.update_coarse_displacement_substep(0.0,0.0)

        # displacement update fine-scale
        self.fine_scale_solver.update_fine_displacement_substep(0.0,0.0)

        # operator-split iterations for sub-step
        max_error = 1.0

        while (max_error > self.TOL):
            
            # acceleration fine-scale update
            self.fine_scale_solver.update_fine_acceleration_substep()

            # acceleration coarse-scale update
            self.coarse_scale_solver.update_coarse_acceleration_substep()

            ## estimate error ##
            max_error_coarse = np.linalg.norm(self.data_manger.a_c_it - self.data_manger.a_c)
            max_error_fine = -1e50
            for unit_cell in range(self.femgeometry.mesh.nel_patch):
                max_error_fine = max(max_error_fine, np.linalg.norm(self.data_manger.a_f_it[unit_cell,:] - self.data_manger.a_f[unit_cell,:]))

            max_error = max(max_error_coarse, max_error_fine)

            # update acceleration iterates
            self.data_manger.a_c = self.data_manger.a_c_it.copy()
            self.data_manger.a_f = self.data_manger.a_f_it.copy()