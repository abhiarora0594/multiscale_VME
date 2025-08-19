import numpy as np
from femgeometry import FEMGeometry
from constitutive_law import Constitutive_Law

# Data Manager Class
class Data_Manager:

    def __init__(self, femgeomtery:FEMGeometry):
        self.femgeomtery = femgeomtery

        # coarse-scale fields before a time step
        self.u_c = np.zeros((self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes.shape[0],1))
        self.v_c = np.zeros((self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes.shape[0],1))
        self.a_c = np.zeros((self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes.shape[0],1))

        # coarse-scale fields after a sub-step
        self.u_c_sub = np.zeros((self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes.shape[0],1))
        self.v_c_sub = np.zeros((self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes.shape[0],1))
        self.a_c_sub = np.zeros((self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes.shape[0],1))

        # coarse-scale fields after a full step
        self.u_c_up = np.zeros((self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes.shape[0],1))
        self.v_c_up = np.zeros((self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes.shape[0],1))
        self.a_c_up = np.zeros((self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes.shape[0],1))

        # coarse-scale acceleration fields during an iteration
        self.a_c_it = np.zeros((self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes.shape[0],1))

        # fine-scale fields before a time step
        self.u_f = np.zeros((self.femgeomtery.mesh.nodes_m.shape[0],self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes_m.shape[1]))
        self.v_f = np.zeros((self.femgeomtery.mesh.nodes_m.shape[0],self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes_m.shape[1]))
        self.a_f = np.zeros((self.femgeomtery.mesh.nodes_m.shape[0],self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes_m.shape[1]))

        # fine-scale fields after a sub-step
        self.u_f_sub = np.zeros((self.femgeomtery.mesh.nodes_m.shape[0],self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes_m.shape[1]))
        self.v_f_sub = np.zeros((self.femgeomtery.mesh.nodes_m.shape[0],self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes_m.shape[1]))
        self.a_f_sub = np.zeros((self.femgeomtery.mesh.nodes_m.shape[0],self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes_m.shape[1]))

        # fine-scale fields after a full step
        self.u_f_up = np.zeros((self.femgeomtery.mesh.nodes_m.shape[0],self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes_m.shape[1]))
        self.v_f_up = np.zeros((self.femgeomtery.mesh.nodes_m.shape[0],self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes_m.shape[1]))
        self.a_f_up = np.zeros((self.femgeomtery.mesh.nodes_m.shape[0],self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes_m.shape[1]))

        # fine-scale acceleration fields during an iteration
        self.a_f_it = np.zeros((self.femgeomtery.mesh.nodes_m.shape[0],self.femgeomtery.dof_per_node*self.femgeomtery.mesh.nodes_m.shape[1]))

    def error_estimate_substep(self):

        max_error_coarse = np.linalg.norm(self.a_c_it - self.a_c_sub)
        max_error_fine = -1e50
        for unit_cell in range(self.femgeometry.mesh.nel_patch):
            max_error_fine = max(max_error_fine, np.linalg.norm(self.a_f_it[unit_cell,:] - self.a_f_sub[unit_cell,:]))

        max_error = max(max_error_coarse, max_error_fine)

        return max_error
    
    def error_estimate_fullstep(self):

        max_error_coarse = np.linalg.norm(self.a_c_it - self.a_c_up)
        max_error_fine = -1e50
        for unit_cell in range(self.femgeometry.mesh.nel_patch):
            max_error_fine = max(max_error_fine, np.linalg.norm(self.a_f_it[unit_cell,:] - self.a_f_up[unit_cell,:]))

        max_error = max(max_error_coarse, max_error_fine)
    
        return max_error

class CoarseScaleSolver:
    
    def __init__(self, femgeomtery:FEMGeometry, data:Data_Manager, constitutive_law:Constitutive_Law):
        self.femgeomtery = femgeomtery
        self.data = data
        self.constitutive_law = constitutive_law

        # Identity tensor in a single column
        self.Id_tensor = np.zeros((4,1)) #[(0,0), (0,1), (1,0), (1,1)]
        self.Id_tensor[0] = 1
        self.Id_tensor[-1] = 1

        # construct the global mass (lumped matrix) for coarse-scale
        self.get_global_mass_matrix_coarse()
    
    def update_coarse_displacement_substep(self,const_v,const_a):
        self.data.u_c_sub = self.data.u_c + np.multiply(self.data.v_c,const_v) + np.multiply(self.data.a_c, const_a)

    def update_coarse_displacement_fullstep(self,const_v,const_a):
        self.data.u_c_up = self.data.u_c_sub + np.multiply(self.data.v_c_sub,const_v) + np.multiply(self.data.a_c_sub, const_a)

    def update_coarse_velocity_substep(self,const_a):
        self.data.v_c_sub = self.data.v_c + np.multiply(const_a,self.data.a_c + self.data.a_c_sub)
    
    def update_coarse_velocity_fullstep(self,const_a0,const_asub,const_aup):
        self.data.v_c_up = self.data.v_c_sub + np.multiply(const_a0,self.data.a_c) + np.multiply(const_asub,self.data.a_c_sub) \
                        + np.multiply(const_aup,self.data.a_c_up)
        
    # global mass matrix for coarse scale
    def get_global_mass_matrix_coarse(self):
        
        # total nodal degree of freedoms
        ndof = len(self.data.u_c)
        m,n = self.femgeomtery.mesh.el_list_patch.shape

        # global internal force vector
        self.Mg = np.zeros((ndof,1))

        for unit_cell_no in range(m):
            for macro_el_no in range(n):
                ie_macro = self.femgeomtery.mesh.el_list_patch[unit_cell_no,macro_el_no]

                # coarse-scale entities per coarse-element 
                X_c_e = self.femgeomtery.mesh.nodes[self.femgeomtery.mesh.elements[ie_macro,:],:]

                # fine-scale entities per coarse-element
                X_f = self.femgeomtery.mesh.nodes_m[unit_cell_no]

                # element level calculations
                Me = self.elem_macroscale_mass_vector(X_c_e,X_f,macro_el_no)

                # assembly
                self.Mg[self.femgeomtery.dof_conn_coarse[ie_macro,:],0] += Me
    

    # acceleration update substep
    def update_coarse_acceleration_substep(self):

        # total nodal degree of freedoms
        ndof = len(self.data.u_c_sub)
        m,n = self.femgeomtery.mesh.el_list_patch.shape

        # global internal force vector
        Fg_int = np.zeros((ndof,1))

        for unit_cell_no in range(m):
            for macro_el_no in range(n):
                ie_macro = self.femgeomtery.mesh.el_list_patch[unit_cell_no,macro_el_no]
                
                # coarse-scale entities per coarse-element 
                u_c_e = self.data.u_c_sub[self.femgeomtery.dof_conn_coarse[ie_macro,:]]
                X_c_e = self.femgeomtery.mesh.nodes[self.femgeomtery.mesh.elements[ie_macro,:],:]

                # fine-scale entities per coarse-element
                u_f = self.data.u_f_sub[unit_cell_no,:]
                a_f = self.data.a_f_it[unit_cell_no,:]
                X_f = self.femgeomtery.mesh.nodes_m[unit_cell_no]

                # element level calculations
                Fe_int = self.elem_macroscale_force_vector(u_c_e,X_c_e,u_f,a_f,X_f,macro_el_no)

                # assembly
                Fg_int[self.femgeomtery.dof_conn_coarse[ie_macro,:]] += Fe_int

        # update the iterate value
        self.data.a_c_it  = np.divide(np.multiply(-1.0,Fg_int), self.Mg)
    
    
    # acceleration update full step
    def update_coarse_acceleration_fullstep(self):

        # total nodal degree of freedoms
        ndof = len(self.data.u_c_up)
        m,n = self.femgeomtery.mesh.el_list_patch.shape

        # global internal force vector
        Fg_int = np.zeros((ndof,1))

        for unit_cell_no in range(m):
            for macro_el_no in range(n):
                ie_macro = self.femgeomtery.mesh.el_list_patch[unit_cell_no,macro_el_no]
                
                # coarse-scale entities per coarse-element 
                u_c_e = self.data.u_c_up[self.femgeomtery.dof_conn_coarse[ie_macro,:]]
                X_c_e = self.femgeomtery.mesh.nodes[self.femgeomtery.mesh.elements[ie_macro,:],:]

                # fine-scale entities per coarse-element
                u_f = self.data.u_f_up[unit_cell_no,:]
                a_f = self.data.a_f_it[unit_cell_no,:]
                X_f = self.femgeomtery.mesh.nodes_m[unit_cell_no]

                # element level calculations
                Fe_int = self.elem_macroscale_force_vector(u_c_e,X_c_e,u_f,a_f,X_f,macro_el_no)

                # assembly
                Fg_int[self.femgeomtery.dof_conn_coarse[ie_macro,:]] += Fe_int

        # update the iterate value
        self.data.a_c_it  = np.divide(np.multiply(-1.0,Fg_int), self.Mg)
    
    # elemenet level force calculation for coarse scale
    def elem_macroscale_force_vector(self, u_c_e, X_c_e, u_f, a_f, X_f, macro_el_no):
        
        # force vector
        Fe_int = np.zeros((self.femgeomtery.dof_per_node*self.femgeomtery.mesh.elements.shape[1],1))

        # B matrix coarse scale
        B_mat_c = np.zeros((4,self.femgeomtery.dof_conn_coarse.shape[1]))

        # B matrix fine scale
        B_mat_f = np.zeros((4,self.femgeomtery.dof_conn_fine.shape[1]))
        
        # loop over the fine-scale elements per coarse element within a patch
        for ie_micro in self.femgeomtery.mesh.el_list_fine_scale_elems_within_each_coarse_elem[macro_el_no]:

            # fine-scale entities
            u_f_e = u_f[self.femgeomtery.dof_conn_fine[ie_micro,:]]
            a_f_e = a_f[self.femgeomtery.dof_conn_fine[ie_micro,:]]
            X_f_e = X_f[self.femgeomtery.mesh.elements_m[ie_micro, :],:]

            # loop over gps
            for gp in range(self.femgeomtery.ngp):

                #xi and eta at a gp
                xi = self.femgeomtery.gp_list[gp,0]
                eta = self.femgeomtery.gp_list[gp,1]

                # x and y coordinates of the Gauss point of micro-element
                X_g = X_f_e.T @ self.femgeomtery.shape_fn.evaluate(xi,eta) # returns 2 by 1

                # natural coordinates in macro element of micro gauss point
                xi_c, eta_c = self.femgeomtery.inverse_coarse_mapping(X_g,X_c_e)

                # get macro shape function gradients at micro gauss points
                [dN_c_global, ] = self.femgeomtery.get_global_shape_gradients(X_c_e,xi_c,eta_c)

                # get micro shape function gradients and Jacobian at micro gauss points
                [dN_f_global, J_f] = self.femgeomtery.get_global_shape_gradients(X_f_e,xi,eta)

                # get macro shape functions at micro gauss points
                N_c = self.femgeomtery.shape_fn.evaluate(xi_c,eta_c)

                # get micro shape functions at micro gauss points
                N_f = self.femgeomtery.shape_fn.evaluate(xi,eta)

                # B matrix course scale is (4 x 8) or (4 x 18)
                B_mat_c[0,0:2:] = dN_c_global[:,0].T # dNci_dx
                B_mat_c[1,0:2:] = dN_c_global[:,1].T # dNci_dy
                B_mat_c[2,1:2:] = dN_c_global[:,0].T # dNci_dx
                B_mat_c[3,1:2:] = dN_c_global[:,1].T # dNci_dy

                # B matrix fine scale is (4 x 8) or (4 x 18)
                B_mat_f[0,0:2:] = dN_f_global[:,0].T # dNfi_dx
                B_mat_f[1,0:2:] = dN_f_global[:,1].T # dNfi_dy
                B_mat_f[2,1:2:] = dN_f_global[:,0].T # dNfi_dx
                B_mat_f[3,1:2:] = dN_f_global[:,1].T # dNfi_dy

                # Deformation Gradient Tensor in a single column  
                # components order -- >[(0,0), (0,1), (1,0), (1,1)]
                F_def = self.Id_tensor + np.matmul(B_mat_c,u_c_e) + np.matmul(B_mat_f,u_f_e)

                # Get the First Piola Kirchhoff stress 
                if self.constitutive_law.material_model == "Neo-Hookean":
                    First_Piola = self.constitutive_law.get_first_piola_stress(F_def, \
                                self.constitutive_law.lambda_list[ie_micro],self.constitutive_law.mu_list[ie_micro])
                else:
                    raise ValueError("only Neo-Hookean model is implemented as a constitutive law")
                
                # element - force vector - stress term
                Fe_int[0:2:,1] = Fe_int[0:2:,1] + np.multiply(J_f*self.femgeomtery.wt_list[gp] \
                                                              , np.matmul(dN_c_global,First_Piola[0:2,1]))
                
                Fe_int[1:2:,1] = Fe_int[1:2:,1] + np.multiply(J_f*self.femgeomtery.wt_list[gp] \
                                                              , np.matmul(dN_c_global,First_Piola[2:,1]))

                # element - force vector - microacceleration term
                Fe_int[0:2:,1] = Fe_int[0:2:,1] + np.multiply(J_f*self.femgeomtery.wt_list[gp]*self.constitutive_law.rho_list[ie_micro] \
                                                            , np.multiply(N_c , np.dot(N_f,a_f_e[0:2:,1])))
                
                Fe_int[1:2:,1] = Fe_int[1:2:,1] + np.multiply(J_f*self.femgeomtery.wt_list[gp]*self.constitutive_law.rho_list[ie_micro] \
                                                            , np.multiply(N_c , np.dot(N_f,a_f_e[1:2:,1])))
                
        return Fe_int
    
    # elemenet level mass vector calculation for coarse scale
    def elem_macroscale_mass_vector(self, X_c_e, X_f, macro_el_no):

        dof_per_elem = self.femgeomtery.dof_per_node*self.femgeomtery.mesh.elements.shape[1]
        
        Me = np.zeros((dof_per_elem,dof_per_elem))

        # loop over the fine-scale elements per coarse element within a patch
        # no_of_micro_elem_per_macro = len(self.femgeomtery.mesh.el_list_fine_scale_elems_within_each_coarse_elem[macro_el_no])
        
        for ie_micro in self.femgeomtery.mesh.el_list_fine_scale_elems_within_each_coarse_elem[macro_el_no]:

            # fine-scale entities
            X_f_e = X_f[self.femgeomtery.mesh.elements_m[ie_micro, :],:]

            # loop over gps
            for gp in range(self.femgeomtery.ngp):

                #xi and eta at a gp
                xi = self.femgeomtery.gp_list[gp,0]
                eta = self.femgeomtery.gp_list[gp,1]

                # x and y coordinates of the Gauss point of micro-element
                X_g = X_f_e.T @ self.femgeomtery.shape_fn.evaluate(xi,eta) # returns 2 by 1

                # natural coordinates in macro element of micro gauss point
                xi_c, eta_c = self.femgeomtery.inverse_coarse_mapping(X_g,X_c_e)

                # get micro shape function gradients and Jacobian at micro gauss points
                [dN_f_global, J_f] = self.femgeomtery.get_global_shape_gradients(X_f_e,xi,eta)

                # get macro shape functions at micro gauss points
                N_c = self.femgeomtery.shape_fn.evaluate(xi_c,eta_c)

                # element - mass vector
                Me_mat = np.zeros((dof_per_elem,dof_per_elem))
                Me_mat[0:2:,0:2:] = np.matmul(N_c,N_c.T)
                Me_mat[1:2:,1:2:] = np.matmul(N_c,N_c.T)

                Me = Me + np.multiply(J_f*self.femgeomtery.wt_list[gp]*self.constitutive_law.rho_list[ie_micro], Me_mat) 
                
        return np.sum(Me,axis=1)

# fine scale class solver
class FineScaleSolver:
    
    def __init__(self, femgeomtery:FEMGeometry, data:Data_Manager, constitutive_law:Constitutive_Law):
        self.femgeomtery = femgeomtery
        self.data = data
        self.constitutive_law = constitutive_law
        
        # Identity tensor in a single column
        self.Id_tensor = np.zeros((4,1))
        self.Id_tensor[0] = 1
        self.Id_tensor[-1] = 1

        # construct the global mass (lumped matrix) for fine-scale
        self.get_global_mass_matrix_fine()
    
    def update_fine_displacement_substep(self,const_v,const_a):
        self.data.u_f_sub = self.data.u_f + np.multiply(self.data.v_f,const_v) + np.multiply(self.data.a_f, const_a)

    def update_fine_displacement_fullstep(self,const_v,const_a):
        self.data.u_f_up = self.data.u_f_sub + np.multiply(self.data.v_f_sub,const_v) + np.multiply(self.data.a_f_sub, const_a)
    
    def update_fine_velocity_substep(self,const_a):
        self.data.v_f_sub = self.data.v_f + np.multiply(const_a,self.data.a_f + self.data.a_f_sub)
    
    def update_fine_velocity_fullstep(self,const_a0,const_asub,const_aup):
        self.data.v_f_up = self.data.v_f_sub + np.multiply(const_a0,self.data.a_f) + np.multiply(const_asub,self.data.a_f_sub) \
                        + np.multiply(const_aup,self.data.a_f_up)


    # mass matrix fine scale
    def get_global_mass_matrix_fine(self):
        
        # total nodal degree of freedoms for 1st unit cell
        ndof = self.data.u_f.shape[1]

        # initializaiton
        self.Mg = np.zeros((ndof,1))

        # fine-scale entities for 1st unit-cell
        X_f = self.femgeomtery.mesh.nodes_m[0]

        # loop over micro-scale elements
        for ie_micro in range(self.femgeomtery.mesh.elements_m.shape[0]):
            
            # fine-scale element level entitites
            X_f_e = X_f[self.femgeomtery.mesh.elements_m[ie_micro,:],:]

            # element level force vector calculation
            Me = self.elem_microscale_mass_vector(X_f_e,ie_micro)
            # print(Me.shape)

            # assembly
            self.Mg[self.femgeomtery.dof_conn_fine[ie_micro,:],0] += Me

    # acceleration update substep
    def update_fine_acceleration_substep(self):

        # total nodal degree of freedoms per unit cell
        ndof = self.data.u_f_sub.shape[1]

        macro_elms = self.femgeomtery.mesh.el_list_fine_scale_elems_within_each_coarse_elem.shape[0]

        for unit_cell_no in range(self.femgeomtery.mesh.nel_patch):

            # global internal force vector
            Fg_int = np.zeros((ndof,1))

            # fine-scale entities per unit-cell
            u_f = self.data.u_f_sub[unit_cell_no,:]
            X_f = self.femgeomtery.mesh.nodes_m[unit_cell_no]

            # for loop over fine-scale elements
            for macro_elem_no in range(macro_elms):

                # coarse-scale entities  
                u_c_e = self.data.u_c_sub[self.femgeomtery.dof_conn_coarse[self.femgeomtery.mesh.el_list_patch[unit_cell_no,macro_elem_no],:]]
                a_c_e = self.data.a_c_it[self.femgeomtery.dof_conn_coarse[self.femgeomtery.mesh.el_list_patch[unit_cell_no,macro_elem_no],:]]
                X_c_e = self.femgeomtery.mesh.nodes[self.femgeomtery.mesh.elements[self.femgeomtery.mesh.el_list_patch[unit_cell_no,macro_elem_no],:],:]

                for ie_micro in self.femgeomtery.mesh.el_list_fine_scale_elems_within_each_coarse_elem[macro_elem_no]:
                    
                    # fine-scale element level entitites
                    u_f_e = u_f[self.femgeomtery.dof_conn_fine[ie_micro,:]]
                    X_f_e = X_f[self.femgeomtery.mesh.elements_m[ie_micro,:],:]

                    # element level force vector calculation
                    Fe_int = self.elem_microscale_force_vector(u_c_e,a_c_e,X_c_e,u_f_e,X_f_e, ie_micro)

                    # assembly
                    Fg_int[self.femgeomtery.dof_conn_fine[ie_micro,:],0] += Fe_int

            # update the iterate value
            self.data.a_f_it[unit_cell_no,:]  = np.divide(np.multiply(-1.0,Fg_int), self.Mg)

    # acceleration update fullstep
    def update_fine_acceleration_fullstep(self):

        # total nodal degree of freedoms per unit cell
        ndof = self.data.u_f_up.shape[1]

        macro_elms = self.femgeomtery.mesh.el_list_fine_scale_elems_within_each_coarse_elem.shape[0]

        for unit_cell_no in range(self.femgeomtery.mesh.nel_patch):

            # global internal force vector
            Fg_int = np.zeros((ndof,1))

            # fine-scale entities per unit-cell
            u_f = self.data.u_f_up[unit_cell_no,:]
            X_f = self.femgeomtery.mesh.nodes_m[unit_cell_no]

            # for loop over fine-scale elements
            for macro_elem_no in range(macro_elms):

                # coarse-scale entities  
                u_c_e = self.data.u_c_up[self.femgeomtery.dof_conn_coarse[self.femgeomtery.mesh.el_list_patch[unit_cell_no,macro_elem_no],:]]
                a_c_e = self.data.a_c_it[self.femgeomtery.dof_conn_coarse[self.femgeomtery.mesh.el_list_patch[unit_cell_no,macro_elem_no],:]]
                X_c_e = self.femgeomtery.mesh.nodes[self.femgeomtery.mesh.elements[self.femgeomtery.mesh.el_list_patch[unit_cell_no,macro_elem_no],:],:]

                for ie_micro in self.femgeomtery.mesh.el_list_fine_scale_elems_within_each_coarse_elem[macro_elem_no]:
                    
                    # fine-scale element level entitites
                    u_f_e = u_f[self.femgeomtery.dof_conn_fine[ie_micro,:]]
                    X_f_e = X_f[self.femgeomtery.mesh.elements_m[ie_micro,:],:]

                    # element level forve vector calculation
                    Fe_int = self.elem_microscale_force_vector(u_c_e,a_c_e,X_c_e,u_f_e,X_f_e, ie_micro)

                    # assembly
                    Fg_int[self.femgeomtery.dof_conn_fine[ie_micro,:],0] += Fe_int

            # update the iterate value
            self.data.a_f_it[unit_cell_no,:]  = np.divide(np.multiply(-1.0,Fg_int), self.Mg)

    # element level force vector microscale calculation
    def elem_microscale_force_vector(self,u_c_e,a_c_e,X_c_e,u_f_e,X_f_e,ie_micro):
        
        # force vector
        Fe_int = np.zeros((self.femgeomtery.dof_per_node*self.femgeomtery.mesh.elements_m.shape[1],1))

        # B matrix coarse scale
        B_mat_c = np.zeros((4,self.femgeomtery.dof_conn_coarse.shape[1]))

        # B matrix fine scale
        B_mat_f = np.zeros((4,self.femgeomtery.dof_conn_fine.shape[1]))

        # loop over gps
        for gp in range(self.femgeomtery.ngp):

            #xi and eta at a gp
            xi = self.femgeomtery.gp_list[gp,0]
            eta = self.femgeomtery.gp_list[gp,1]

            # x and y coordinates of the Gauss point of micro-element
            X_g = X_f_e.T @ self.femgeomtery.shape_fn.evaluate(xi,eta) # returns 2 by 1

            # natural coordinates in macro element of micro gauss point
            xi_c, eta_c = self.femgeomtery.inverse_coarse_mapping(X_g,X_c_e)

            # get macro shape function gradients at micro gauss points
            [dN_c_global, ] = self.femgeomtery.get_global_shape_gradients(X_c_e,xi_c,eta_c)

            # get micro shape function gradients and Jacobian at micro gauss points
            [dN_f_global, J_f] = self.femgeomtery.get_global_shape_gradients(X_f_e,xi,eta)

            # get macro shape functions at micro gauss points
            N_c = self.femgeomtery.shape_fn.evaluate(xi_c,eta_c)

            # get micro shape functions at micro gauss points
            N_f = self.femgeomtery.shape_fn.evaluate(xi,eta)

            # B matrix course scale is (4 x 8) or (4 x 18)
            B_mat_c[0,0:2:] = dN_c_global[:,0].T # dNci_dx
            B_mat_c[1,0:2:] = dN_c_global[:,1].T # dNci_dy
            B_mat_c[2,1:2:] = dN_c_global[:,0].T # dNci_dx
            B_mat_c[3,1:2:] = dN_c_global[:,1].T # dNci_dy

            # B matrix fine scale is (4 x 8) or (4 x 18)
            B_mat_f[0,0:2:] = dN_f_global[:,0].T # dNfi_dx
            B_mat_f[1,0:2:] = dN_f_global[:,1].T # dNfi_dy
            B_mat_f[2,1:2:] = dN_f_global[:,0].T # dNfi_dx
            B_mat_f[3,1:2:] = dN_f_global[:,1].T # dNfi_dy

            # Deformation Gradient Tensor in a single column
            # components order -- >[(0,0), (0,1), (1,0), (1,1)]
            F_def = self.Id_tensor + np.matmul(B_mat_c,u_c_e) + np.matmul(B_mat_f,u_f_e)

            # Get the First Piola Kirchhoff stress 
            if self.constitutive_law.material_model == "Neo-Hookean":
                First_Piola = self.constitutive_law.get_first_piola_stress(F_def, \
                            self.constitutive_law.lambda_list[ie_micro],self.constitutive_law.mu_list[ie_micro])
            else:
                raise ValueError("only Neo-Hookean model is implemented as a constitutive law")
            
            # element - force vector - stress term
            Fe_int[0:2:,1] = Fe_int[0:2:,1] + np.multiply(J_f*self.femgeomtery.wt_list[gp] \
                                                            , np.matmul(dN_f_global,First_Piola[0:2,1]))
            
            Fe_int[1:2:,1] = Fe_int[1:2:,1] + np.multiply(J_f*self.femgeomtery.wt_list[gp] \
                                                            , np.matmul(dN_f_global,First_Piola[2:,1]))

            # element - force vector - macroacceleration term
            Fe_int[0:2:,1] = Fe_int[0:2:,1] + np.multiply(J_f*self.femgeomtery.wt_list[gp]*self.constitutive_law.rho_list[ie_micro] \
                                                        , np.multiply(N_f , np.dot(N_c,a_c_e[0:2:,1])))
            
            Fe_int[1:2:,1] = Fe_int[1:2:,1] + np.multiply(J_f*self.femgeomtery.wt_list[gp]*self.constitutive_law.rho_list[ie_micro] \
                                                        , np.multiply(N_f , np.dot(N_c,a_c_e[1:2:,1])))

        return Fe_int
    
    # elemenet level mass vector calculation for fine scale
    def elem_microscale_mass_vector(self, X_f_e, ie_micro):

        dof_per_elem = self.femgeomtery.dof_per_node*self.femgeomtery.mesh.elements_m.shape[1]
        
        Me = np.zeros((dof_per_elem,dof_per_elem))

        # loop over gps
        for gp in range(self.femgeomtery.ngp):

            #xi and eta at a gp
            xi = self.femgeomtery.gp_list[gp,0]
            eta = self.femgeomtery.gp_list[gp,1]

            # get micro shape function gradients and Jacobian at micro gauss points
            [dN_f_global, J_f] = self.femgeomtery.get_global_shape_gradients(X_f_e,xi,eta)

            # get macro shape functions at micro gauss points
            N_f = self.femgeomtery.shape_fn.evaluate(xi,eta)

            # element - mass vector
            Me_mat = np.zeros((dof_per_elem,dof_per_elem))
            Me_mat[0:2:,0:2:] = np.matmul(N_f,N_f.T)
            Me_mat[1:2:,1:2:] = np.matmul(N_f,N_f.T)

            Me = Me + np.multiply(J_f*self.femgeomtery.wt_list[gp]*self.constitutive_law.rho_list[ie_micro], Me_mat) 
                
        return np.sum(Me,axis=1)