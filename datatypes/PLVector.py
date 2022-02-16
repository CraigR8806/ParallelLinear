from typing import Any
import numpy as np
from  plinear.datatypes.Matrix import Matrix
import plinear.calculations.ParallelLinear as pl



class Vector(Matrix):

    def __init__(self, lengthOrVals, **kwargs):
        super().__init__(1, lengthOrVals, **kwargs)
           



    def setAtPos(self, x, val):
        self.data[x] = val
    
    def getAtPos(self, x):
        return self.data[x]

    def __str__(self):
        return str(self.data)

    def exportToList(self):
        return self.data.tolist()

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
            return Vector(self.getNumberOfRows(), pl._addScaler(self, a))

    def subScaler(self, a, in_place = True) -> Any:
        if in_place:
            pl._subScalerInPlace(self, a)
        else:
            return Vector(self.getNumberOfRows(), pl._subScaler(self, a))

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