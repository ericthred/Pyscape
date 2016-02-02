import numpy as np
# import seaborn as sns
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF



# x1 = [1, 2, 3, 5, 6]
# y1 = [1, 4.5, 7, 24, 38]
#
# # (2) Make dictionary linking x and y coordinate lists to 'x' and 'y' keys
# trace1 = dict(x=x1, y=y1)
#
# # (3) Make list of 1 trace, to be sent to Plotly
# data = [trace1]
#
#
# # (@) Call the plot() function of the plotly.plotly submodule,
# #     save figure as 's0_first_plot'
# py.plot(data, filename='s0_first_plot')


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
    global count
    base_levels = rec[rec == range(nn)]
    stack = np.zeros(nn, dtype=int)
    count = 0

    def add_to_stack(ijk):
        global count
        stack[count] = ijk
        count += 1
        for donor in donors[ijk,:ndon[ijk]]:
            if donor != -1:
                add_to_stack(donor)

    for base in base_levels:
        add_to_stack(base)

    return stack


def pland(h_in):
    # cmap = sns.cubehelix_palette(8)
    ii = range(nx)
    jj = range(ny)
    zz = h_in.reshape(ny,nx)
    # annotate plot with i and j
    annotations = []
    for n, row in enumerate(zz):
        for m, val in enumerate(row):
            var = zz[n][m]
            annotations.append(
                dict(
                    text=str(val),
                    x=np.round(ii[m]), y=np.round(jj[n]),
                    xref='x1', yref='y1',
                    font=dict(color='black'),
                    showarrow=False)
                )

    trace = [go.Heatmap(x=ii, y=jj, z=zz, colorscale='RdBu', showscale=False)]

    fig = go.Figure(data=trace)
    fig['layout'].update(
        title="Annotated Heatmap",
        # annotations=annotations,
        xaxis=dict(ticks='', side='top'),
        # ticksuffix is a workaround to add a bit of padding
        yaxis=dict(ticks='', ticksuffix='  '),
        width=1000*nx/max([nx,ny]),
        height=1000*ny/max([nx,ny]),
        autosize=False
    )

    py.plot(fig, filename='labelled_heatmap')


def v_plot(h_in, d_in, s_in):
    # set up x and y arrow vectors
    U = np.zeros(nn) # x vector
    V = np.zeros(nn) # y vector
    # assign the proper vector direction based on the recieving neighbour
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

    # make a grid of the x and y locations of all the points
    qx, qy = np.meshgrid(   np.arange(nx)*dx + dx/2.,
                            np.arange(ny)*dy + dy/2.)
    # reshape the U and V arrow vectors into a grid
    qU = (dx*U).reshape(ny,nx)
    qV = (dy*V).reshape(ny,nx)

    # create a plotly quiver figure using the vectors I made
    fig = FF.create_quiver(qx, qy, qU, qV,
                       scale=.9,
                       arrow_scale=.1,
                       name='quiver',
                       line=dict(width=1))

    # identify all the base points and pits
    base_levels = rec[rec == range(nn)]

    # create x and y locations for those base points
    x2 = np.mod(base_levels, nx)*dx + dx/2.
    y2 = np.floor(base_levels/nx)*dy + dy/2.
    # create a plotly scatter plot object using those locations
    points = go.Scatter(x=x2, y=y2,
                    mode='markers',
                    marker=dict(size=12),
                    name='points')

    # Add points to quiver figure
    fig['data'].append(points)

    # send to plotly
    py.plot(fig, filename='Quiver with Points')

    # ii = range(nx)
    # jj = range(ny)
    # zz = h_in.reshape(ny,nx)
    # # annotate plot with i and j
    # annotations = []
    # for n, row in enumerate(zz):
    #     for m, val in enumerate(row):
    #         var = zz[n][m]
    #         annotations.append(
    #             dict(
    #                 text=str(val),
    #                 x=ii[m], y=jj[n],
    #                 xref='x1', yref='y1',
    #                 font=dict(color='black'),
    #                 showarrow=False)
    #             )
    #
    # trace = [go.Heatmap(x=ii, y=jj, z=zz, colorscale='RdBu', showscale=False)]
    # fig['layout'].update(
    #     title="Quiver Heatmap",
    #     annotations=annotations,
    #     xaxis=dict(ticks='', side='top'),
    #     # ticksuffix is a workaround to add a bit of padding
    #     yaxis=dict(ticks='', ticksuffix='  '),
    #     width=1000,
    #     height=500,
    #     autosize=False
    # )
    # add data to figure
    # fig['data'].append(trace)
    # Q = plt.quiver(qx,qy,qU,qV, scale=max([xl,yl]))
    # plt.axis('equal')
    # plt.show()


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

# # test grid
# h = np.array([9,0,0,0,6,6,6,5,4,3,
#               2,2,2,2,5,5,5,4,4,2,
#               3,3,3,3,5,4,3,2,1,0,
#               2,2,2,2,5,5,5,4,4,2,
#               0,0,0,0,6,6,6,5,4,3])
# nx, ny = (10, 5)
# xl, yl = (10, 5) # meters
# dx, dy = (xl/(nx-1), yl/(ny-1))
# dd = np.sqrt(dx**2 + dy**2)
# nn = nx*ny

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
pland(ndon)

# calculate the stack
stack = make_stack()
print 'stack:\n', np.reshape(stack, (ny,nx))
pland(stack)

# v_plot(h, direction, vector)
