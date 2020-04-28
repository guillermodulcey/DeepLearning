import os
import cv2
from skimage import color
import numpy as np
import matplotlib.pyplot as plt

from Cache.Management.Management import Management
from Filtros.FilterManager import FilterManager
from Filtros.Filter import FilterFactory
from Activacion.Activation import ActivationFactory
from Pooling.Pooling import Pooling
from Entrenamiento.DataOperations import DataOperations

class Coach():
    def __init__(self):
        super().__init__()
    
    def createTrainingPool(self,perro,filtersList,biasList,poolType,sizeAndStride):
        filtros = filtersList
        biases = biasList
        path = f'2020_Training_Test/{perro}/train'
        archivos = os.listdir(path)

        ic = ImageConverter()

        for archivo in archivos:
            image = ic.resizeImage(path,archivo)
            result = ic.convertImage(image,perro,archivo,filtros,biases,poolType,sizeAndStride)
            self.saveVector(perro,result)

    def saveVector(self, perro, npArray):
        result = np.array(npArray).flatten()
        if perro == 'beagle':
            perroClass = 1
        if perro == 'boxer':
            perroClass = 2
        if perro == 'chihuahua':
            perroClass = 3
        result = np.append(result,perroClass)
        result = ' '.join(map(str, list(result))).replace(' ',',')
        f = open('Vectors/training.txt','a')
        f.write(f'{result}\n')
        f.close()

class Overseer():
    def __init__(self):
        super().__init__()
        self.do = DataOperations()

    def learnTraining(self):
        from sklearn.preprocessing import MinMaxScaler
        data = self.do.cargarDatos('Vectors/training.txt')
        x,y = self.do.obtenerXY(data)
        # x = MinMaxScaler().fit_transform(x)
        clasificador = Clasificador().getClasificador('MLP',100)
        self.model = clasificador.fit(x,y)

    def testTraining(self):
        from sklearn.metrics import confusion_matrix
        from sklearn import metrics
        data = self.do.cargarDatos('Vectors/test.txt')
        x,right = self.do.obtenerXY(data)
        result = self.model.predict(x)
        confusion_matrix = confusion_matrix(right, result)
        accuracy = metrics.accuracy_score(right, result)
        return confusion_matrix, accuracy

    def convertTestData(self,perro,filtros,biases,poolType,sizeAndStride):
        path = f'2020_Training_Test/{perro}/test'
        archivos = os.listdir(path)

        ic = ImageConverter()
        for archivo in archivos:
            image = ic.resizeImage(path,archivo)
            result = ic.convertImage(image,perro,archivo,filtros,biases,poolType,sizeAndStride)
            self.saveVector(perro,result,archivo)

    def saveVector(self, perro, npArray, archivo):
        result = np.array(npArray).flatten()
        if perro == 'beagle':
            perroClass = 1
        if perro == 'boxer':
            perroClass = 2
        if perro == 'chihuahua':
            perroClass = 3
        result = np.append(result,perroClass)
        result = ' '.join(map(str, list(result))).replace(' ',',')
        f = open('Vectors/test.txt','a')
        f.write(f'{result}\n')
        f.close()
        f = open('Vectors/file_list.txt','a')
        f.write(f'{archivo}\n')
        f.close()

class ImageConverter():
    def __init__(self):
        super().__init__()

    def convertImage(self,image,perro,archivo,filtros,biases,poolType,sizeAndStride):
        filterResult = image
        
        # Cálculo de filtros
        ################################################################################
        for filtro in filtros:
            parameters_filter = {'NAME':archivo, 'FILTER':filtro[0], 'FILTER_NAME':filtro[1]}
            cache_filter = Management(parameters_filter) 
            if cache_filter.isInCache('FILTER'):
                filterResult = cache_filter.loadFromCache('FILTER')
            else:
                if parameters_filter['FILTER'] == 'CUSTOM':
                    filterManager = FilterManager()
                    kernel = filterManager.loadFilter(parameters_filter['FILTER_NAME'])
                    filterResult = FilterFactory().getFilter(parameters_filter["FILTER"]).convolute(filterResult,kernel)
                else:
                    filterResult = FilterFactory().getFilter(parameters_filter["FILTER"]).convolute(filterResult,parameters_filter['FILTER_NAME'])
                cache_filter.saveInCache(filterResult,'FILTER')
            # self.showImage(filterResult)
        ################################################################################
        for bias in biases:
            # Cálculo de bias
            ################################################################################
            parameters_activation = parameters_filter.copy()
            parameters_activation.update({'ACTIVATION':'RELU', 'BIAS':bias})

            cache_activation = Management(parameters_activation)

            if cache_activation.isInCache('ACTIVATION'):
                biasResult = cache_activation.loadFromCache('ACTIVATION')
            else:
                biasResult = ActivationFactory().getActivation(parameters_activation).applyActivation(filterResult)
                cache_activation.saveInCache(biasResult,'ACTIVATION')
            # self.showImage(biasResult)
        ################################################################################
        # Cálculo de Pooling
        result = biasResult
        for i in range(len(poolType)):
            size = sizeAndStride[i][0]
            stride = sizeAndStride[i][1]
            result = Pooling(poolType[i]).poolImage(result,size,stride)
            # self.showImage(result)
        return result
        ################################################################################
    
    def resizeImage(self,path,archivo):
        image = cv2.imread(f'{path}/{archivo}', cv2.IMREAD_COLOR)
        image = color.rgb2gray(image)

        new_width = 300
        new_height = 300

        dim_size = (new_width,new_height)

        image = cv2.resize(image, dim_size, interpolation = cv2.INTER_AREA)
        return image

    def showImage(self,image):
        plt.imshow(image, cmap=plt.cm.gray)
        plt.axis('off')
        plt.show()

class Clasificador():
    def __init__(self):
        super().__init__()

    def getClasificador(self,name,itera):
        if name == 'MLP':
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(max_iter=itera)
        if name == 'SVM':
            from sklearn.svm import SVC
            return SVC(kernel='linear',max_iter=itera)

    