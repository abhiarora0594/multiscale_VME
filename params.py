import numpy as np

CFL =  1.0 # CFL for the Noh and Bathe's scheme
P_ = 0.54 # substep ratio
T_TOTAL = 0.3 # total time
TOL = 1e-4 # VME iterations tolerance
MATERIAL_MODEL = "Neo-Hookean"
MU = 1.0
LAMBDA = 1.0
RHO = 1.0
NSTEPS = 100