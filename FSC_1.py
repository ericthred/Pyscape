import numpy as np, matplotlib.pyplot as plt, seaborn as sns
sns.set(style='darkgrid')
sns.set_context("paper")
plt.close()


def boundary_conditions(BC):
    if BC == 0:
        def make_nbor(xx):
            # main points
            if xx == 0:
                def nbor_list(ij):
                    return [ij-nx-1,ij-nx,ij-nx+1,ij-1,ij+1,ij+nx-1,ij+nx,ij+nx+1]
            # left side
            if xx == 1:
                def nbor_list(ij):
                    return [ij,ij-nx,ij-nx+1,ij,ij+1,ij,ij+nx,ij+nx+1]
            # right side
            if xx == 2:
                def nbor_list(ij):
                    return [ij-nx-1,ij-nx,ij,ij-1,ij,ij+nx-1,ij+nx,ij]
            # bottom side
            if xx == 3:
                def nbor_list(ij):
                    return [ij,ij,ij,ij-1,ij+1,ij+nx-1,ij+nx,ij+nx+1]
            # top side
            if xx == 4:
                def nbor_list(ij):
                    return [ij-nx-1,ij-nx,ij-nx+1,ij-1,ij+1,ij,ij,ij]
            # bottom left corner
            if xx == 5:
                def nbor_list(ij):
                    return [ij,ij,ij,ij,ij+1,ij,ij+nx,ij+nx+1]
            # bottom right corner
            if xx == 6:
                def nbor_list(ij):
                    return [ij,ij,ij,ij-1,ij,ij+nx-1,ij+nx,ij]
            # top left corner
            if xx == 7:
                def nbor_list(ij):
                    return [ij,ij-nx,ij-nx+1,ij,ij+1,ij,ij,ij]
            # top right corner
            if xx == 8:
                def nbor_list(ij):
                    return [ij-nx-1,ij-nx,ij,ij-1,ij,ij,ij,ij]

            return nbor_list
        return make_nbor


def receiver():
    rec = np.arange(nn) # receiver array
    vector = np.zeros(nn) # vector array
    direction = 4*np.ones(nn) # direction array
    len_nbor = [dd, dy, dd, dx, dx, dd, dy, dd] # distance to each neighbour
    loc_nbor = [0, 1, 2, 3, 5, 6, 7, 8]
    ii_ranges = [range(1,nx-1), [0], [nx-1], range(1,nx-1), range(1,nx-1), [0], [nx-1], [0], [nx-1]]
    jj_ranges = [range(1,ny-1), range(1,ny-1), range(1,ny-1), [0], [ny-1], [0], [0], [ny-1], [ny-1]]

    for xx in range(9):
        nbor_list = make_nbor(xx)
        for jj in jj_ranges[xx]:
            for ii in ii_ranges[xx]:
                ij = ii + jj*nx # linear index
                nbor = nbor_list(ij) # neighbours
                slopes = (h[ij] - h[nbor]) / len_nbor # slope between neighbours and node ij
                smax = 0.
                for ind, slope in enumerate(slopes): # iterate over neighbours
                    if slope > smax: # find max slope
                        smax = slope # save new max slope
                        vector[ij] = slope
                        rec[ij] = nbor[ind]
                        direction[ij] = loc_nbor[ind]

    return rec, vector, direction


def donor_list():
    ndon = np.zeros(nn, dtype=int)
    donor = -1*np.ones((nn,8), dtype=int)
    for ij in range(nn):
        if rec[ij] != ij:
            ijk = rec[ij]
            donor[ijk, ndon[ijk]] = ij
            ndon[ijk] += 1

    return donor, ndon


def make_stack():
    # make nstack global so it works in recursive algy
    global nstack
    # find all baselevel points
    base_levels = rec[rec == range(nn)]
    # initialise stack
    stack = np.zeros(nn, dtype=int)
    colour = np.zeros(nn, dtype=int)
    # initialise stack counter
    nstack = 0

    # define recursive stack function
    def add_to_stack(ijk, c_flag):
        # make nstack global so it works in recursive algy
        global nstack
        # but node on the stack
        stack[nstack]  = ijk
        colour[ijk] = c_flag
        # increase counter by one
        nstack += 1

        # iterate over nodes donors
        for donor in donors[ijk,:ndon[ijk]]:
            # start process over by calling stack algy on donor
            add_to_stack(donor, c_flag)

    # call recursive algy on all base level points
    c_count = 0
    for base in base_levels:
        add_to_stack(base, c_count)
        c_count += 1

    # return the stack
    return stack, colour


def calculate_area():
    # reverse the stack
    rstack = np.flipud(stack)
    # iterate over the stack, adding the area
    A = np.ones(nn) * dx * dy
    for ij in rstack:
        if rec[ij] != ij:
            A[rec[ij]] += A[ij]

    return A


def pland(h_in):
    cmap = sns.cubehelix_palette(8, as_cmap=True)
    plt.pcolormesh(h_in.reshape(ny, nx, order='C'), cmap=cmap)
    # plt.contourf(h_in.reshape(ny, nx, order='C'), cmap=cmap)
    plt.axis('equal')
    plt.colorbar()
    plt.show()


def v_plot(h_in, d_in, s_in):
    fig = plt.figure(1)

    h_in = h_in.reshape(ny, nx, order='C')
    cmap = sns.cubehelix_palette(8, as_cmap=True, dark=0.3)
    plt.pcolormesh(h_in.reshape(ny, nx, order='C'), cmap=cmap)
    plt.colorbar()
    # plt.contourf(h_in.reshape(ny, nx, order='C'), cmap=cmap)
    s_in = s_in/s_in.max()
    U = np.zeros(nn)
    V = np.zeros(nn)
    for ij in range(nn):
        if d_in[ij] == 0:
            U[ij] = -1
            V[ij] = -1
        if d_in[ij] == 1:
            U[ij] = 0
            V[ij] = -1
        if d_in[ij] == 2:
            U[ij] = 1
            V[ij] = -1
        if d_in[ij] == 3:
            U[ij] = -1
            V[ij] = 0
        if d_in[ij] == 4:
            U[ij] = 0
            V[ij] = 0
        if d_in[ij] == 5:
            U[ij] = 1
            V[ij] = 0
        if d_in[ij] == 6:
            U[ij] = -1
            V[ij] = 1
        if d_in[ij] == 7:
            U[ij] = 0
            V[ij] = 1
        if d_in[ij] == 8:
            U[ij] = 1
            V[ij] = 1

    qx = np.arange(nx)*dx + dx/2.
    qy = np.arange(ny)*dy + dy/2.
    qU = (dx*U).reshape(ny,nx)
    qV = (dy*V).reshape(ny,nx)
    Q = plt.quiver(qx,qy,qU,qV, scale=max([xl,yl])*2.)
    plt.axis('equal')
    plt.xlim([0, nx])
    plt.ylim([0, ny])
    plt.show()


# set the scale of the grid
xl, yl = (100.e3, 100.e3) # meters

# set the resolution of the grid
nx, ny = (31, 51)
dx, dy = (xl/(nx-1), yl/(ny-1))
dd = np.sqrt(dx**2 + dy**2)
nn = nx*ny

# set the timestep vector
dt = 1000. # years

# number of timesteps
nstep = 1000

# set the parameters of the stream power law
n = 1.
m = n*0.4

# initial conditions
h = np.random.rand(nn)

# test grid
h = np.array([9,0,0,0,6,6,6,5,4,3,
              2,2,2,2,5,5,5,4,4,2,
              3,3,3,3,5,4,3,2,1,0,
              2,2,2,2,5,5,5,4,4,2,
              0,0,0,0,6,6,6,5,4,3])
nx, ny = (10, 5)
xl, yl = (10, 5) # meters
dx, dy = (xl/(nx-1), yl/(ny-1))
dd = np.sqrt(dx**2 + dy**2)
nn = nx*ny

# pland(h)

# boundary conditions
# all boundaries are base level
make_nbor = boundary_conditions(0)
# h = 0 at y = 0 and yl, cyclic at x = 0 and xl
# make_nbor = boundary_conditions(1)
# h = 0 at y = 0 and yl, reflective at x = 0 and xl
# make_nbor = boundary_conditions(2)

# calculate the receiver array
rec, vector, direction = receiver()
# print 'receiver:\n', np.reshape(rec, (ny,nx))
# pland(rec)

# calculate the donor array
donors, ndon = donor_list()
# print 'ndon:\n', np.reshape(ndon, (ny,nx))
# pland(ndon)

# calculate the stack
stack, colour = make_stack()
# print 'stack:\n', np.reshape(stack, (ny,nx))
# pland(stack)
# pland(colour)

# calculate the catchment area for each node
A = calculate_area()

v_plot(h, direction, vector)
v_plot(colour, direction, vector)
v_plot(A, direction, vector)
