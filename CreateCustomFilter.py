import numpy as np
from Filtros.FilterManager import FilterManager

# Existen dos formas de crear un filtro:
# 
# A partir de una matriz:
matriz = [[0.0625, 0.125, 0.0625],[0.125, 0.25, 0.125],[0.0625, 0.125, 0.0625]]
FilterManager().createFilter('blur',np.array(matriz))

matriz = [[-1, -2, -1],[0, 0, 0],[1, 2, 1]]
FilterManager().createFilter('bsobel',np.array(matriz))

matriz = [[-2, -1, 0],[-1, 1, 1],[0, 1, 2]]
FilterManager().createFilter('emboss',np.array(matriz))

matriz = [[0, 0, 0],[0, 1, 0],[0, 0, 0]]
FilterManager().createFilter('iden',np.array(matriz))

matriz = [[1, 0, -1],[2, 0, -2],[1, 0, -1]]
FilterManager().createFilter('lsobel',np.array(matriz))

matriz = [[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]]
FilterManager().createFilter('outli',np.array(matriz))

matriz = [[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]
FilterManager().createFilter('rsobel',np.array(matriz))

matriz = [[0, -1, 0],[-1, 5, -1],[0, -1, 0]]
FilterManager().createFilter('sharp',np.array(matriz))

matriz = [[1, 2, 1],[0, 0, 0],[-1, -2, -1]]
FilterManager().createFilter('tsobel',np.array(matriz))

# A partir de una imagen
#  
# El primer parámetro es el nombre de la imagen
# El segundo el nombre del archivo resultante (para su uso como parámetro en Prueba.py):
# FilterManager().createFilterFromImage('hocicobea.PNG','bea1')
# FilterManager().createFilterFromImage('ojobea.PNG','bea2')
# FilterManager().createFilterFromImage('pechobea.PNG','bea3')

# FilterManager().createFilterFromImage('hocicobox.PNG','box1')
# FilterManager().createFilterFromImage('hocicobox2.PNG','box2')
# FilterManager().createFilterFromImage('ojobox.PNG','box3')
# FilterManager().createFilterFromImage('orejabox.PNG','box4')
# FilterManager().createFilterFromImage('orejabox2.PNG','box5')

# FilterManager().createFilterFromImage('hocicochi.PNG','chi1')
# FilterManager().createFilterFromImage('orejachi.PNG','chi2')
# FilterManager().createFilterFromImage('orejachi2.PNG','chi3')