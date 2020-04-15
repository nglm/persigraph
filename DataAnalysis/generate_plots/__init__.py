# ------
# Source
# https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# ------

import gen_figs_all_members_first_location