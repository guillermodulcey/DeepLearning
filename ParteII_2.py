from Entrenamiento.Coach import Coach
from Entrenamiento.Coach import Overseer

from Cache.Management.Management import Management

# opciones: beagle boxer chihuahua
perros = ['beagle','boxer','chihuahua']
###################################
# opciones: SOBEL GAUSSIAN LAPLACE UNIFORM PREWITT (para PREDETERMINED)
# opciones: bea1 bea2 bea3 box1 box2 box3 box4 box5 chi1 chi2 chi3 blur bsobel emboss iden lsobel outli rsobel sharp tsobel(para CUSTOM)
filterList = [('PREDETERMINED','SOBEL')]
###################################
# opciones: cualquier valor real
biasList = [0]
###################################
poolType = ['MAX','AVERAGE','MAX','AVERAGE']
# opciones: cualquier valor entero positivo
sizeAndStride = [(3,2),(3,2),(2,1),(2,1)]
###################################
Management().emptyCache()
co = Coach()
ov = Overseer()

# for perro in perros:
#     co.createTrainingPool(perro,filterList,biasList,poolType,sizeAndStride)
#     ov.convertTestData(perro,filterList,biasList,poolType,sizeAndStride)

ov.learnTraining()
confusion_matrix,accuracy = ov.testTraining()

print(confusion_matrix)
print(f'Accuracy: {accuracy}')