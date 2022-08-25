from ..vis.commonstyle import get_list_colors

def draw_arrow(
    g,
    ax,
    k,
    t_start,
    t_end,
    offset_arrow = +5,
    offset_text = +10,
    width = 5,
    txt_kw = {'size':18},
    color_list = None,
):
    if color_list is None:
        color_list = get_list_colors(g.N)
    x_start = (
        (g.time_axis[t_start] - g.time_axis[0])
        / (g.time_axis[-1] - g.time_axis[0])
    )
    x_end = (
        (g.time_axis[t_end] - g.time_axis[0])
        / (g.time_axis[-1] - g.time_axis[0])
    )
    x_mid = (x_end + x_start) / 2

    if t_start == t_end:
        arrowstyle = '|-|, widthA=.5, widthB=.5'
    else:
        arrowstyle = '|-|'

    # Arrow
    ax.annotate(
        '',
        xy=(x_start, offset_arrow),
        xycoords=('axes fraction', 'axes points'),
        #xytext=(x_end+0.001, 0),
        xytext=(x_end, 0),
        textcoords=('axes fraction', 'offset points'),
        arrowprops=dict(arrowstyle=arrowstyle, color=color_list[k]),
    )
    # Thicker Arrow
    ax.annotate(
        '',
        xy=(x_start+0.0025, offset_arrow),
        xycoords=('axes fraction', 'axes points'),
        #xytext=(x_end+0.001, 0),
        xytext=(x_end-0.0025, 0),
        textcoords=('axes fraction', 'offset points'),
        arrowprops=dict(
            width=width, headwidth=0, headlength=0.0001, color=color_list[k]
        ),
    )

    # Text
    ax.annotate(
        k,
        xy=(x_mid, offset_text),
        xycoords=('axes fraction', 'axes points'),
        xytext=(-4, 0),
        textcoords=('offset points', 'offset points'),
        color=color_list[k], **txt_kw
    )