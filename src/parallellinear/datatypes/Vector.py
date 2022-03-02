from typing import Any
from parallellinear.calculations.LinearCalculations import LinearCalculations
import numpy as np
from parallellinear.datatypes.Matrix import Matrix



class Vector(Matrix):

    def __init__(self, data, calcManager=None):
        super().__init__(1, data, calcManager)
           

    @classmethod
    def random(cls, length:int, calcManager = Matrix.CALCULATIONS_MANAGER, random_low=0, random_high=1):
        if calcManager==None:
            calcManager = Matrix.CALCULATIONS_MANAGER
        out = cls(data=np.random.rand(length).astype(calcManager.getPrecision()), calcManager=calcManager)
        if random_low != 0 or random_high != 1:
            out.scale(random_high-random_low)
            out.addScaler(random_low)
        return out 

    @classmethod
    def fromList(cls, data:list, calcManager = Matrix.CALCULATIONS_MANAGER):
        if calcManager==None:
            calcManager = Matrix.CALCULATIONS_MANAGER
        return cls(data=np.array(data, dtype=calcManager.getPrecision()), calcManager=calcManager)

    
    @classmethod
    def zeros(cls, length:int, calcManager = Matrix.CALCULATIONS_MANAGER):
        if calcManager==None:
            calcManager = Matrix.CALCULATIONS_MANAGER
        return cls(data=np.zeros(length, dtype=calcManager.getPrecision()), calcManager=calcManager)

    @classmethod
    def filledWithValue(cls, length:int, value, calcManager = Matrix.CALCULATIONS_MANAGER):
        return cls(data=np.empty(length, dtype=calcManager.getPrecision()).fill(value), calcManager=calcManager)



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
            self.calcManager.getCalculator()._addInPlace(self.data, a.getData())
        else:
            return Vector(self.calcManager.getCalculator()._add(self.data, a.getData()), self.calcManager)

    def sub(self, a, in_place = True) -> Any:
        if in_place:
            self.calcManager.getCalculator()._subInPlace(self.data, a.getData())
        else:
            return Vector(self.calcManager.getCalculator()._sub(self.data, a.data), self.calcManager)

    def addScaler(self, scaler, in_place = True) -> Any:
        if in_place:
            self.calcManager.getCalculator()._addScalerInPlace(self.data, scaler)
        else:
            return Vector(self.calcManager.getCalculator()._addScaler(self.data, scaler), self.calcManager)

    def subScaler(self, scaler, in_place = True) -> Any:
        if in_place:
            self.calcManager.getCalculator()._subScalerInPlace(self.data, scaler)
        else:
            return Vector(self.calcManager.getCalculator()._subScaler(self.data, scaler), self.calcManager)

    def subScalerFrom(self, scaler, in_place = True) -> Any:
        if in_place:
            self.calcManager.getCalculator()._subScalerFromInPlace(self.data, scaler)
        else:
            return Vector(self.calcManager.getCalculator()._subScalerFrom(self.data, scaler), self.calcManager)

    def scale(self, scaler, in_place = True) -> Any:
        if in_place:
            self.calcManager.getCalculator()._scaleInPlace(self.data, scaler)
        else:
            return Vector(self.calcManager.getCalculator()._scale(self.data, scaler), self.calcManager)

    def descale(self, scaler, in_place = True) -> Any:
        if in_place:
            self.calcManager.getCalculator()._descaleInPlace(self.data, scaler)
        else:
            return Vector(self.calcManager.getCalculator()._descale(self.data, scaler), self.calcManager)

    def applyCustomFunction(self, func_name, in_place = True):
        if in_place:
            self.calcManager.getCalculator()._applyCustomFunctionInPlace(self.data, func_name)
        else:
            return Vector(self.calcManager.getCalculator()._applyCustomFunction(self.data, func_name), self.calcManager)

    def magnitude(a):
        return np.sqrt(np.sum((lambda x:np.square(x))(a)))

    def elementWiseMultiply(self, a, in_place = True):
        if in_place:
            self.calcManager.getCalculator()._elementWiseMultiplyInPlace(self.data, a.getData())
        else:
            return Vector(self.calcManager.getCalculator()._elementWiseMultiply(self.data, a.getData()), self.calcManager)