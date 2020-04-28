import math
import random

class DataOperations():
    def obtenerX(self, dataSet: list):
        x = []
        for i in range(len(dataSet)):
            vX = []
            for j in range(len(dataSet[i])-1):
                vX.append(dataSet[i][j])
            x.append(vX)
        return x

    def obtenerY(self, dataSet: list):
        y = []
        for i in range(len(dataSet)):
                y.append(dataSet[i][-1])
        return y

    def obtenerXY(self, dataSet: list):
        x = []
        y = []
        for i in range(len(dataSet)):
            vX = []
            for j in range(len(dataSet[i])-1):
                vX.append(dataSet[i][j])
            x.append(vX)
            y.append(dataSet[i][-1])
        return x, y

    #Solo funciona con datasets de dos clases
    def obtenerDataSets(self, dataSet: list, proporcion: float, seed=1):
        dataSetAux = dataSet.copy()
        cantidad = len(dataSet)
        random.seed(seed)

        dataSetEntrenamiento = []
        entrenamiento = math.floor(cantidad * proporcion)
        clase = 0

        while len(dataSetEntrenamiento) < entrenamiento:
            posicion = random.randint(0,len(dataSetAux)-1)
            registro = dataSetAux[posicion].copy()

            if clase == registro[-1]:
                dataSetEntrenamiento.append(registro)
                dataSetAux.pop(posicion)
                clase = abs(clase-1)

        return dataSetEntrenamiento, dataSetAux

    def cargarDatos(self, fileName: str, separador = ','):
        f = open(fileName,'r')

        dataSet = []
        for x in f:
            registro = x.rstrip().split(separador)
            dataSet.append(list(map(float,registro)))

        f.close()

        return dataSet