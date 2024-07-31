import numpy as np

U = np.loadtxt("../data/sod/U.txt")
g = np.loadtxt("../data/sod/g.txt")
# U: | t | x | rho | u | T (on the third order gauss points)
# g: | t | x | v | g (on the third order gauss points)
