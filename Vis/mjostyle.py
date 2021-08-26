from matplotlib.collections import EllipseCollection


def draw_mjo_classes(
    fig=None,
    ax=None,
    kw : dict = {}
):

    if

    # Weak VS strong MJO
    circle = EllipseCollection(
        widths = 2,
        heights = 2,
        angles = 0,
        units = 'xy',
        edgecolors = 'lightgrey',
        facecolors = "none",
        offsets = (0, 0),
        transOffset = ax.transData)
    ax.add_collection(circle)

    # Create the data for the 4 lines
    N_points = 1000
    x1 = np.linspace(-4,4,N_points)
    y1 = np.array([x for x in x1])
    y2 = np.array([0 for x in x1])
    y3 = np.array([-x for x in x1])
    y4 = np.array([x for x in x1])
    x4 = np.zeros(N_points)

    # Now let's remove the parts that are inside the circle
    cond1 = np.sqrt(x1**2+y1**2) >= 1
    cond2 = np.sqrt(x1**2+y2**2) >= 1
    cond3 = np.sqrt(x1**2+y3**2) >= 1
    cond4 = np.sqrt(x4**2+y4**2) >= 1

    x1_filt = x1[cond1]
    y1_filt = y1[cond1]

    x2_filt = x1[cond2]
    y2_filt = y2[cond2]

    x3_filt = x1[cond3]
    y3_filt = y3[cond3]

    x4_filt = x4[cond4]
    y4_filt = y4[cond4]

    # Plot lines
    ax.plot(x1_filt[x1_filt<0], y1_filt[x1_filt<0], lw=0.6, c='lightgrey')
    ax.plot(x1_filt[x1_filt>0], y1_filt[x1_filt>0], lw=0.6, c='lightgrey')
    ax.plot(x2_filt[x2_filt<0], y2_filt[x2_filt<0], lw=0.6, c='lightgrey')
    ax.plot(x2_filt[x2_filt>0], y2_filt[x2_filt>0], lw=0.6, c='lightgrey')
    ax.plot(x3_filt[x3_filt<0], y3_filt[x3_filt<0], lw=0.6, c='lightgrey')
    ax.plot(x3_filt[x3_filt>0], y3_filt[x3_filt>0], lw=0.6, c='lightgrey')
    ax.plot(x4_filt[y4_filt<0], y4_filt[y4_filt<0], lw=0.6, c='lightgrey')
    ax.plot(x4_filt[y4_filt>0], y4_filt[y4_filt>0], lw=0.6, c='lightgrey')
    return fig, ax
