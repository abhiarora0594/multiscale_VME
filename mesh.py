import numpy as np
import math

# mesh class
class Mesh2D:
	
    # constructor
    def __init__(self, nx, ny, lx = 1.0, ly = 1.0, element_type="Q1", no_of_elems_in_each_patch = 1, nx_f = 10, ny_f = 10):
        """
        Parameters:
            nx, ny        : Number of elements in x and y
            lx, ly        : Physical size in x and y
            element_type  : "Q1" (4-node) or "Q2" (9-node)
        """
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.element_type = element_type.upper()

        if self.element_type == "Q1":
            self.nnx = nx + 1
            self.nny = ny + 1
        elif self.element_type == "Q2":
            self.nnx = 2 * nx + 1
            self.nny = 2 * ny + 1
        else:
            raise ValueError("element_type must be 'Q1' or 'Q2'")
        
        # some global variables (coarse)
        self.nel = self.nx*self.ny
        self.no_of_elems_in_each_patch = no_of_elems_in_each_patch

        if self.nel % self.no_of_elems_in_each_patch == 0:
            self.nel_patch = int(self.nel/self.no_of_elems_in_each_patch)
        else:
            raise ValueError("Total no of elems not divisible by no of elemens in each patch")

        # generate coarse nodes, elements, and list of coarse patches
        self.nodes = self._generate_nodes()
        self.elements, self.boundary_elements = self._generate_elements()
        self.el_list_patch = self._generate_list_of_coarse_patches()

        # fine scale elements
        self.nx_f = nx_f
        self.ny_f = ny_f
        self.nel_f = nx_f*ny_f

        if self.element_type == "Q1":
            self.nnx_f = self.nx_f + 1
            self.nny_f = self.ny_f + 1
        elif self.element_type == "Q2":
            self.nnx_f = 2 * self.nx_f + 1
            self.nny_f = 2 * self.ny_f + 1
        else:
            raise ValueError("element_type must be 'Q1' or 'Q2'")
        
        # generate fine scale nodes, elements, and list of fine elements per coarse patch
        self.nodes_m = self._generate_fine_scale_nodes()
        self.elements_m, self.boundary_elements_m = self._generate_fine_scale_elements()
        self.el_list_fine_scale_elems_within_each_coarse_elem = self._generate_fine_scale_elements_within_each_coarse_elem()
        
    # class member functions
    def _generate_nodes(self):
        x = np.linspace(0, self.lx, self.nnx)
        y = np.linspace(0, self.ly, self.nny)
        xv, yv = np.meshgrid(x, y, indexing='ij')
        coords = np.column_stack((xv.ravel(), yv.ravel()))

        return coords

    def _generate_elements(self):
        
        elements = []
        boundary_elements = []

        if self.element_type == "Q1":

            for i in range(self.nx):
                for j in range(self.ny):
                    n0 = i * self.nny + j # bottom-left
                    n1 = n0 + 1 # top-left
                    n2 = n0 + self.nny # bottom-right
                    n3 = n2 + 1 # top-right

                    # 2-d elements
                    elements.append([n0, n2, n3, n1])  # Q1: ccw

                    # Only keep edges on boundary
                    if j == 0:          # bottom
                        boundary_elements.append([n0, n2])  # bottom-left -> bottom-right
                    if i == self.nx-1:  # right
                        boundary_elements.append([n2, n3])  # bottom-right -> top-right
                    if j == self.ny-1:  # top
                        boundary_elements.append([n3, n1])  # top-right -> top-left
                    if i == 0:          # left
                        boundary_elements.append([n1, n0])  # top-left -> bottom-left


        elif self.element_type == "Q2":

            for i in range(self.nx):
                for j in range(self.ny):
                    i0 = 2 * i
                    j0 = 2 * j
                    n = lambda ii, jj: (i0 + ii) * self.nny + (j0 + jj)

                    #2-d elements
                    elements.append([
                        n(0, 0),  # bottom-left
                        n(2, 0),  # bottom-right
                        n(2, 2),  # top-right
                        n(0, 2),  # top-left
                        n(1, 0),  # bottom-mid
                        n(2, 1),  # right-mid
                        n(1, 2),  # top-mid
                        n(0, 1),  # left-mid
                        n(1, 1)  # center
                    ])

                    # Bottom edge
                    if j == 0:
                        boundary_elements.append([n(0,0), n(1,0), n(2,0)])
                    # Right edge
                    if i == self.nx-1:
                        boundary_elements.append([n(2,0), n(2,1), n(2,2)])
                    # Top edge
                    if j == self.ny-1:
                        boundary_elements.append([n(2,2), n(1,2), n(0,2)])
                    # Left edge
                    if i == 0:
                        boundary_elements.append([n(0,2), n(0,1), n(0,0)])

        return np.array(elements, dtype = np.int64), np.array(boundary_elements, dtype = np.int64)
    
    def get_element_coordinates(self, eid):
        # print(self.elements[eid])
        return self.nodes[self.elements[eid]]

    def print_summary(self):
        print(f"{self.element_type} Mesh: {self.nx} x {self.ny} elements")
        print(f"Nodes: {len(self.nodes)}, Elements: {len(self.elements)}")
    
    def _generate_list_of_coarse_patches(self):
        
        # el_list_patch = np.zeros((self.nel_patch,self.no_of_elems_in_each_patch))
        el_list_patch = [[] for i in range(self.nel_patch)]

        if (self.no_of_elems_in_each_patch != 1):
            if (self.no_of_elems_in_each_patch - math.isqrt(self.no_of_elems_in_each_patch)**2 !=0):
                raise ValueError("No of elements in each coarse patch should be perfect square or equal to 1")
        
        if (self.no_of_elems_in_each_patch == 1):
            nelms_in_each_dir_patch = 1
        else:
            nelms_in_each_dir_patch = int(np.sqrt(self.no_of_elems_in_each_patch))

        llx = (self.lx/self.nx)*nelms_in_each_dir_patch
        lly = (self.ly/self.ny)*nelms_in_each_dir_patch

        for i in range(int(self.nx/nelms_in_each_dir_patch)):
            for j in range(int(self.ny/nelms_in_each_dir_patch)):

                ind = i*int(self.ny/nelms_in_each_dir_patch) +  j

                el_lx = i*llx
                el_rx = (i+1)*llx
                el_by = j*lly
                el_ty =  (j+1)*lly

                # print(self.elements)

                for elem in range(len(self.elements)):
                    if (self.element_type=="Q1"):
                        x_c = np.mean(self.get_element_coordinates(elem),axis = 0)[0]
                        y_c = np.mean(self.get_element_coordinates(elem),axis = 0)[1]
                    elif (self.element_type=="Q2"):
                        x_c = self.get_element_coordinates(elem)[-1,0]
                        y_c = self.get_element_coordinates(elem)[-1,1]
                    else:
                        raise ValueError("element_type must be 'Q1' or 'Q2'")

                    if ( el_lx < x_c and x_c < el_rx and el_by < y_c  and y_c < el_ty):
                        el_list_patch[ind].append(elem)
                        continue                 

        return np.array(el_list_patch, dtype = np.int64)

    def _generate_fine_scale_nodes(self):

        if (self.no_of_elems_in_each_patch != 1):
            if (self.no_of_elems_in_each_patch - math.isqrt(self.no_of_elems_in_each_patch)**2 !=0):
                raise ValueError("No of elements in each coarse patch should be perfect square or equal to 1")
        
        if (self.no_of_elems_in_each_patch == 1):
            nelms_in_each_dir_patch = 1
        else:
            nelms_in_each_dir_patch = int(np.sqrt(self.no_of_elems_in_each_patch))

        llx = (self.lx/self.nx)*nelms_in_each_dir_patch
        lly = (self.ly/self.ny)*nelms_in_each_dir_patch

        # print(f"{llx} coarse patch length along x \n")
        # print(f"{lly} coarse patch length along y \n")

        nodes_micro = []

        for i in range(int(self.nx/nelms_in_each_dir_patch)):
            for j in range(int(self.ny/nelms_in_each_dir_patch)):

                ind = i*int(self.ny/nelms_in_each_dir_patch) +  j

                el_lx = i*llx
                el_rx = (i+1)*llx
                el_by = j*lly
                el_ty =  (j+1)*lly

                x = np.linspace(el_lx, el_rx, self.nnx_f)
                y = np.linspace(el_by, el_ty, self.nny_f)
                xv, yv = np.meshgrid(x, y, indexing='ij')
                coords = np.column_stack((xv.ravel(), yv.ravel()))
                nodes_micro.append(coords)

        return np.array(nodes_micro)
    
    def _generate_fine_scale_elements(self):
        
        elements = []
        boundary_elements = []

        if self.element_type == "Q1":
            for i in range(self.nx_f):
                for j in range(self.ny_f):
                    n0 = i * self.nny_f + j
                    n1 = n0 + 1
                    n2 = n0 + self.nny_f
                    n3 = n2 + 1

                    #2-d elements
                    elements.append([n0, n2, n3, n1])  # Q1: ccw

                    # Only keep edges on boundary
                    if j == 0:          # bottom
                        boundary_elements.append([n0, n2])  # bottom-left -> bottom-right
                    if i == self.nx-1:  # right
                        boundary_elements.append([n2, n3])  # bottom-right -> top-right
                    if j == self.ny-1:  # top
                        boundary_elements.append([n3, n1])  # top-right -> top-left
                    if i == 0:          # left
                        boundary_elements.append([n1, n0])  # top-left -> bottom-left
                

        elif self.element_type == "Q2":
            for i in range(self.nx_f):
                for j in range(self.ny_f):
                    i0 = 2 * i
                    j0 = 2 * j
                    n = lambda ii, jj: (i0 + ii) * self.nny_f + (j0 + jj)

                    # 2-d elements
                    elements.append([
                        n(0, 0),  # bottom-left
                        n(2, 0),  # bottom-right
                        n(2, 2),  # top-right
                        n(0, 2),  # top-left
                        n(1, 0),  # bottom-mid
                        n(2, 1),  # right-mid
                        n(1, 2),  # top-mid
                        n(0, 1),  # left-mid
                        n(1, 1)  # center
                    ])

                    # Bottom edge
                    if j == 0:
                        boundary_elements.append([n(0,0), n(1,0), n(2,0)])
                    # Right edge
                    if i == self.nx-1:
                        boundary_elements.append([n(2,0), n(2,1), n(2,2)])
                    # Top edge
                    if j == self.ny-1:
                        boundary_elements.append([n(2,2), n(1,2), n(0,2)])
                    # Left edge
                    if i == 0:
                        boundary_elements.append([n(0,2), n(0,1), n(0,0)])


        return np.array(elements, dtype = np.int64), np.array(boundary_elements, dtype = np.int64)
    
    def _generate_fine_scale_elements_within_each_coarse_elem(self):

        elements = [[] for i in range(self.no_of_elems_in_each_patch)]

        if (self.no_of_elems_in_each_patch != 1):
            if (self.no_of_elems_in_each_patch - math.isqrt(self.no_of_elems_in_each_patch)**2 !=0):
                raise ValueError("No of elements in each coarse patch should be perfect square or equal to 1")
        
        if (self.no_of_elems_in_each_patch == 1):
            nelms_in_each_dir_patch = 1
        else:
            nelms_in_each_dir_patch = int(np.sqrt(self.no_of_elems_in_each_patch))

        # length in each direction of a coarse elemeent
        llx = (self.lx/self.nx)
        lly = (self.ly/self.ny)

        _nodes_m = self.nodes_m[0]
        # print(_nodes_m.shape)

        # print(np.shape(self.elements_m)[0])

        for elem in range(0,np.shape(self.elements_m)[0]):
            # print(self.elements_m[elem])
            if (self.element_type=="Q1"):
                x_c = np.mean(_nodes_m[self.elements_m[elem]],axis = 0)[0]
                y_c = np.mean(_nodes_m[self.elements_m[elem]],axis = 0)[1]
            elif (self.element_type=="Q2"):
                x_c = _nodes_m[self.elements_m[elem]][-1,0]
                y_c = _nodes_m[self.elements_m[elem]][-1,1]
            else:
                raise ValueError("element_type must be 'Q1' or 'Q2'")
            
            for i in range(nelms_in_each_dir_patch):
                for j in range(nelms_in_each_dir_patch):
                    if (x_c > i*llx and x_c < (i+1)*llx and y_c >j*lly and y_c < (j+1)*lly):
                        # indexing is consistent with el-list patch indexing
                        elements[i*nelms_in_each_dir_patch+j].append(elem)

        return np.array(elements, dtype = np.int64)