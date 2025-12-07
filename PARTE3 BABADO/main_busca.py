import numpy as np
import matplotlib.pyplot as plt
from busca_discreta import *

origem = [2,10]
quantidade_pontos = 10
pontos = np.vstack((
    origem,
    np.random.uniform(-10, 20, size=(quantidade_pontos,2))
))

# grs = GlobalRandomSearch(1000,pontos)
# grs.search()
# grs = LocalRandomSearch(1000,2,pontos)
# grs.search()
grs = simulated_annealing(1000, pontos)
grs.search()