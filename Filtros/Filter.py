# Fábrica
######################################################################
class FilterFactory():
    def __init__(self):
        super().__init__()
    
    def getFilter(self,typ):
        if typ == 'CUSTOM':
            return Custom()
        if typ == 'PREDETERMINED':
            return Predet()

######################################################################

# Clases
######################################################################
class Filter():
    def __init__(self):
        super().__init__()

    def convolute(self, image, filtr):
        return 'instanceless'

class Custom(Filter):
    def __init__(self):
        super().__init__()

    def convolute(self, image, filtr):
        from Filtros.Convolucion import Convolucion
        return Convolucion().convolve2d(image,filtr)

class Predet(Filter):
    def __init__(self):
        super().__init__()

    def convolute(self, image, filtr):
        from scipy import ndimage
        if filtr == 'SOBEL':
            return ndimage.sobel(image)
        if filtr == 'GAUSSIAN':
            return ndimage.gaussian_filter(image,sigma=20)
        if filtr == 'LAPLACE':
            return ndimage.laplace(image)
        if filtr == 'UNIFORM':
            return ndimage.uniform_filter(image)
        if filtr == 'PREWITT':
            return ndimage.prewitt(image)
######################################################################