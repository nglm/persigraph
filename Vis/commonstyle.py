
# See https://colorbrewer2.org/#type=qualitative&scheme=Set1&n=8
COLOR_BREWER = [
    "#636363", # Grey
    "#377eb8", # Blue
    "#a65628", # Brown
    "#984ea3", # Purple
    "#e41a1c", # Red
    "#4daf4a", # Green
    "#ff7f00", # Orange
    "#f781bf", # Pink
    "#ffff33", # Yellow
]

COLOR_BREWER_RGB = [
    np.array(ImageColor.getcolor(c, "RGB"))/255 for c in COLOR_BREWER
    ]
COLOR_BREWER_RGBA = [np.r_[c, np.ones(1)] for c in COLOR_BREWER_RGB]


    def get_list_colors(
        N: int,
    ) -> List:
        """
        Repeat COLOR_BREWER list until we get exactly N colors

        :param N: Number of colors desired
        :type N: int
        :return: List of colors (taken from COLOR_BREWER list)
        :rtype: List
        """
        n_cb = len(COLOR_BREWER)
        list_colors = []
        for i in range(1 + N//n_cb) :
            list_colors += COLOR_BREWER_RGBA
        return list_colors[:(N+1)]