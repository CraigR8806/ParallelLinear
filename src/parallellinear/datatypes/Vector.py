from typing import Any
import numpy as np

import parallellinear.calculations.ParallelLinear as pl
from parallellinear.datatypes.Matrix import Matrix



class Vector(Matrix):

    def __init__(self, data):
        super().__init__(1, data)
           

    @classmethod
    def random(cls, length:int, random_low=0, random_high=1):
        out = cls(data=np.random.rand(length).astype(np.float32))
        if random_low != 0 or random_high != 1:
            out.scale(random_high-random_low)
            out.addScaler(random_low)
        return out 

    @classmethod
    def fromList(cls, data:list):
        return cls(data=np.array(data).astype(np.float32))

    
    @classmethod
    def zeros(cls, length:int):
        return cls(data=np.zeros(length).astype(np.float32))

    @classmethod
    def filledWithValue(cls, length:int, value):
        return cls(data=np.empty(length).fill(value).astype(np.float32))



    def setAtPos(self, x, val):
        self.data[x] = val
    
    def getAtPos(self, x):
        return self.data[x]

    def __str__(self):
        return str(self.data)

    def exportToList(self):
        return self.data.tolist()[0]

    def getData(self):
        return self.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, indicies):
        return self.data[indicies]


    def add(self, a, in_place = True) -> Any:
        if in_place:
            pl._addInPlace(self, a)
        else:
            return Vector(pl._add(self, a))

    def sub(self, a, in_place = True) -> Any:
        if in_place:
            pl._subInPlace(self, a)
        else:
            return Vector(pl._sub(self, a))

    def addScaler(self, a, in_place = True) -> Any:
        if in_place:
            pl._addScalerInPlace(self, a)
        else:
            return Vector(pl._addScaler(self, a))

    def subScaler(self, a, in_place = True) -> Any:
        if in_place:
            pl._subScalerInPlace(self, a)
        else:
            return Vector(pl._subScaler(self, a))

    def subScalerFrom(self, a, in_place = True) -> Any:
        if in_place:
            pl._subScalerFromInPlace(self, a)
        else:
            return Vector(pl._subScalerFrom(self, a))

    def scale(self, scaler, in_place = True) -> Any:
        if in_place:
            pl._scaleInPlace(self, scaler)
        else:
            return Vector(pl._scale(self, scaler))

    def descale(self, scaler, in_place = True) -> Any:
        if in_place:
            pl._descaleInPlace(self, scaler)
        else:
            return Vector(pl._descale(self, scaler))

    def applyCustomFunction(self, func_name, in_place = True):
        if in_place:
            pl._applyCustomFunctionInPlace(self, func_name)
        else:
            return Vector(pl._applyCustomFunction(self, func_name))

    def magnitude(a):
        return np.sqrt(np.sum((lambda x:np.square(x))(a)))

    def elementWiseMultiply(self, a, in_place = True):
        if in_place:
            pl._elementWiseMultiplyInPlace(self, a)
        else:
            return Vector(pl._elementWiseMultiply(self, a))