import numpy as np
import cv2
from skimage import color
import matplotlib.pyplot as plt

from Pooling.Pooling import Pooling

from Cache.Management.Management import Management
from Filtros.FilterManager import FilterManager
from Filtros.Filter import FilterFactory

parameters_filter = {'NAME':'cat.jpg', 'FILTER':'PREDETERMINED', 'FILTER_NAME':'SOBEL'}
parameters_activation = parameters_filter.copy()
parameters_activation.update({'ACTIVATION':'RELU', 'BIAS':0})

image = cv2.imread(f'Archivos/{parameters_filter["NAME"]}', cv2.IMREAD_COLOR)
image = color.rgb2gray(image)
filterManager = FilterManager()
image = FilterFactory().getFilter(parameters_filter["FILTER"]).convolute(image,parameters_filter['FILTER_NAME'])

size = 3
stride = 2
result = Pooling('MAX').poolImage(image,size,stride)

plt.imshow(result, cmap=plt.cm.gray)
plt.axis('off')
plt.show()