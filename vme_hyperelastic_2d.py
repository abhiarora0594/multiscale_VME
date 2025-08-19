import numpy as np
from mesh import Mesh2D
from femgeometry import FEMGeometry
from constitutive_law import Constitutive_Law
from multiscale_algorithm import MultiScaleAlgorithm
from params import *

# driver script begins here
if __name__ == "__main__":
    
    # generate mesh
    mesh = Mesh2D(nx=10, ny=10, lx=1.0, ly=1.0, element_type="Q1", no_of_elems_in_each_patch = 4, nx_f = 10, ny_f = 10)

    # generate the FEM elements
    femgeometry = FEMGeometry(mesh, dof_per_node = 2, ngp = 4)

    # define constitutive law
    constituive_law = Constitutive_Law(mesh,MATERIAL_MODEL)

    # assign material properties
    if MATERIAL_MODEL == "Neo-Hookean":
        constituive_law.assign_neohookean_properties(MU,LAMBDA,RHO)

    # start the multiscale algorithm
    multiscale_algorithm = MultiScaleAlgorithm(femgeometry,
                                         constituive_law, 
                                         CFL = CFL, 
                                         P_ = P_, 
                                         T_TOTAL= T_TOTAL, 
                                         TOL = TOL,
                                         NSTEPS= NSTEPS)

    mesh.print_summary()

    print("\nMesh nodes:")
    print(femgeometry.mesh.nodes)

    print("\nMesh elements (node indices):")
    print(mesh.elements)

    print("\nCoordinates of first element:")
    print(mesh.get_element_coordinates(3))

    print("\nElement-list coarse patch:")
    print(mesh.el_list_patch)

    print("\nmesh nel patch:")
    print(mesh.nel_patch)

    print("\nFine-scale mesh nodes per coarse patch:")
    print(mesh.nodes_m.shape)

    print("\nFine-scale mesh elements per coarse patch:")
    print(mesh.elements_m.shape)

    print("\nFine-scale mesh elements per coarse element within a coarse patch:")
    print(mesh.el_list_fine_scale_elems_within_each_coarse_elem)

    xi, eta = 0.0, 0.0  # center of reference element
    X_c = np.array([[-2,0],[0,0],[0,2],[-2,2]])
    grads, detJ = femgeometry.get_global_shape_gradients(X_c, xi=xi, eta=eta)

    print("Global gradients at center:\n", grads)
    print("Jacobian determinant:", detJ)

    print(f"q0 is :{multiscale_algorithm.q0}")
    print(f"q1 is :{multiscale_algorithm.q1}")
    print(f"q2 is :{multiscale_algorithm.q2}")

    xi , eta = -1/np.sqrt(3) , -1/np.sqrt(3)
    # xi , eta  = 0.0 , 0.0
    X_f = np.array([[-2,0],[0,0],[0,2],[-2,2]]) 
    X_g = X_f.T @ femgeometry.shape_fn.evaluate(xi,eta)
    print(X_g)
    X_c = np.array([[-2,-2],[2,-2],[2,2],[-2,2]])
    print(X_c.shape)
    xi_c, eta_c = femgeometry.inverse_coarse_mapping(X_g,X_c)
    print(f"xi_c is {xi_c}")
    print(f"eta_c is {eta_c}")

    N_c = femgeometry.shape_fn.evaluate(xi,eta)
    print(f"shape of N_c is {N_c.shape}")

