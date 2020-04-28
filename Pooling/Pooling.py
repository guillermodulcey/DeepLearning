import numpy as np

class Pooling():
    def __init__(self,operator):
        super().__init__()
        self.operator = OperatorFactory().getOperator(operator)

    def poolImage(self,image,fieldSize,stride):
        result = []
        workImage = np.array(image)
        for x in range(workImage.shape[0]):
            rows = []
            if x%fieldSize==0:
                for y in range(workImage.shape[1]):
                    if y%fieldSize==0:
                        matrix = workImage[x:x+fieldSize, y:y+fieldSize]
                        rows.append(self.operator.operate(matrix))
                result.append(rows)
        return result


# Operadores
# Fabrica
class OperatorFactory():
    def __init__(self):
        super().__init__()

    def getOperator(self,name):
        if name == 'MAX':
            return MaxOperator()
        if name == 'AVERAGE':
            return AverageOperator()
# Clase abstracta
########################################
class Operator():
    def __init__(self):
        super().__init__()

    def operate(self,matrix):
        return 'instanceless'
########################################

# Clase abstracta
class AverageOperator(Operator):
    def __init__(self):
        super().__init__()

    def operate(self,matrix):
        return np.mean(matrix)

class MaxOperator(Operator):
    def __init__(self):
        super().__init__()

    def operate(self,matrix):
        return matrix.max()
########################################