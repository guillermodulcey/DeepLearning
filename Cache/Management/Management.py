import hashlib
import numpy as np
import os.path

class Management():
    def __init__(self, identifiers: dict={}):
        super().__init__()
        self.identifiers = identifiers

    def saveInCache(self, result, typ):
        name = self.getName()
        np.save(f'Cache/Archivos/{name}_{typ}',result)

    def loadFromCache(self, typ):
        name = self.getName()
        return np.load(f'Cache/Archivos/{name}_{typ}.npy')

    def getName(self):
        return hashlib.sha256(str(self.identifiers).encode("utf-8")).hexdigest()

    def isInCache(self, typ):
        name = self.getName()
        return os.path.isfile(f'Cache/Archivos/{name}_{typ}.npy')

    def emptyCache(self):
        path = 'Cache/Archivos'
        archivos = os.listdir(path)
        for archivo in archivos:
            # print(f'{path}/{archivo}')
            os.remove(f'{path}/{archivo}')