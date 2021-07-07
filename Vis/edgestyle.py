class EdgeStyle():


    def __edges_lines(g, edges):
        '''
        Define a line representing the edge for each edge in edges

        Return a nested list (nb edges, 1 line)
        '''
        t_start = g.time_axis[edges[0].time_step]
        t_end = g.time_axis[edges[0].time_step + 1]
        lines = [
            (
            (t_start,   e.v_start.info['mean']),
            (t_end,     e.v_end.info['mean'])
            ) for e in edges
        ]
        return lines



    def __std_polygon(g, edges):
        '''
        Define a polygon representing the uncertainty of each edge in edges

        Return a nested list (nb edges, 1 polygon)
        '''
        t_start = g.time_axis[edges[0].time_step]
        t_end = g.time_axis[edges[0].time_step + 1]
        # std_inf(t) - >std_inf(t) -> std_sup(t+1) -> std_inf(t+1)
        polys = [[
            # std_inf at t
            (t_start, e.v_start.info["mean"] - e.v_start.info["std_inf"]),
            # std_sup at t
            (t_start, e.v_start.info["mean"] + e.v_start.info["std_sup"]),
            # std_sup at t+1
            (t_end,   e.v_end.info["mean"] + e.v_end.info["std_sup"]),
            # std_inf at t+1
            (t_end,   e.v_end.info["mean"] - e.v_end.info["std_inf"]),]
            for e in edges
        ]
        return polys