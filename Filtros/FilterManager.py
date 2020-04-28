import cv2
from skimage import color
from PIL import Image
import numpy as np

class FilterManager():
    def __init__(self):
        super().__init__()

    def createFilter(self, name, npArray):
        np.save(f'Filtros/CUSTOM/{name}',npArray)

    def createFilterFromImage(self, image, name):
        result = cv2.imread(f'Archivos/{image}', cv2.IMREAD_GRAYSCALE)
        print(result)
        imageShape = result.shape
        result = np.array(list(map(lambda x: 1 if x > 128 else -1,result.flatten())))
        result = result.reshape(imageShape)

        # import matplotlib.pyplot as plt
        # plt.imshow(result, cmap=plt.cm.gray)
        # plt.axis('off')
        # plt.show()
        # print(result)
        # result = color.rgb2gray(result)

        print(result)
        self.createFilter(name,result)

    def loadFilter(self, name):
        return np.load(f'Filtros/CUSTOM/{name}.npy')
