import cv2
import numpy as np

from skimage import color
from PIL import Image
import matplotlib.pyplot as plt

from Filtros.Filter import FilterFactory
from Activacion.Activation import ActivationFactory
from Cache.Management.Management import Management
from Filtros.FilterManager import FilterManager

# opciones de parametros:
# 
# Parámetros de filtro:
# NAME: cualquier imagen que esté en la carpeta archivos
# FILTER: tiene dos opciones CUSTOM y PREDETERMINED
#           CUSTOM: Filtros basados en matrices, se encuentran en la subcarpeta CUSTOM de la carpeta Filtros
#                   Si se desea generar un filtro se puede hacer uso del script CreateCustomFilter.py
#           PREDETERMINED: Son filtros predeterminados de la librería scipy
# FILTER_NAME: El nombre del filtro, en caso de ser un filtro CUSTOM, poner el nombre con el que se ha guardado (obviar la extensión.npy)
#               En caso de ser PREDETERMINED, las opciones son: SOBEL, GAUSSIAN, LAPLACE, UNIFORM, PREWITT
# 
# Parámetros de activación
# ACTIVATION: Función de activación que se va a aplicar a la imagen convolucionada, las opciones son: RELU
# BIAS: Parámetro para la función de activación RELU
# 
# Este script guarda en "CACHE" los resultados de las operaciones para ahorrar tiempos de ejecución
# Descomentar la siguiente linea si se desea borrar la "CACHE" (Los archivos cache se generan en base a los parámetros utilizados)
# Management().emptyCache()

parameters_filter = {'NAME':'cat.jpg', 'FILTER':'PREDETERMINED', 'FILTER_NAME':'LAPLACE'}
parameters_activation = parameters_filter.copy()
parameters_activation.update({'ACTIVATION':'RELU', 'BIAS':0})

cache_filter = Management(parameters_filter)
cache_activation = Management(parameters_activation)

filterManager = FilterManager()

image = cv2.imread(f'Archivos/{parameters_filter["NAME"]}', cv2.IMREAD_COLOR)
image = color.rgb2gray(image)
print(image)

# Descomentar si se quiere ver la imagen original
# plt.imshow(image, cmap=plt.cm.gray)
# plt.axis('off')
# plt.show()


if cache_filter.isInCache('FILTER'):
    result = cache_filter.loadFromCache('FILTER')
else:
    if parameters_filter['FILTER'] == 'CUSTOM':
        kernel = filterManager.loadFilter(parameters_filter['FILTER_NAME'])
        print('kernel')
        print(kernel)
        result = FilterFactory().getFilter(parameters_filter["FILTER"]).convolute(image,kernel)
    else:
        result = FilterFactory().getFilter(parameters_filter["FILTER"]).convolute(image,parameters_filter['FILTER_NAME'])
    cache_filter.saveInCache(result,'FILTER')

print(result)

# Descomentar si se quiere ver la imagen convolucionada
# plt.imshow(result, cmap=plt.cm.gray)
# plt.axis('off')
# plt.show()

if cache_activation.isInCache('ACTIVATION'):
    result = cache_activation.loadFromCache('ACTIVATION')
else:
    result = ActivationFactory().getActivation(parameters_activation).applyActivation(result)
    cache_activation.saveInCache(result,'ACTIVATION')

print(result)

plt.imshow(result, cmap=plt.cm.gray)
plt.axis('off')
plt.show()