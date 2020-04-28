
import numpy as np

class Convolucion():
    def __init__(self):
        super().__init__()

    def convolve2d(self, image, kernel):
        kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel
        output = np.zeros_like(image)  # convolution output

        ## métodos de generalización ##
        padding = combineShapes(image.shape,kernel.shape)
        coordinates = addSizeShape(kernel.shape,-1)
        ###############################

        # Add zero padding to the input image
        padded = np.zeros(padding)

        padded[coordinates[0]:-coordinates[0], coordinates[1]:-coordinates[1]] = image
        for x in range(image.shape[1]):  # Loop over every pixel of the image
            for y in range(image.shape[0]):
                # if x%4 == 0 and y%4 == 0:
                #     output[y, x] = (kernel * padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]).sum()
                # element-wise multiplication of the kernel and the image
                output[y, x] = (kernel * padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]).sum()
        return output

def combineShapes(image, kernel):
    result = []
    for i in range(len(image)):
        result.append(2*kernel[i]+(image[i]-2))
    return tuple(result)

def addSizeShape(shape, size):
    result = []
    for i in range(len(shape)):
        result.append(shape[i]+size)
    return tuple(result)