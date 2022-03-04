from typing import Any
import numpy as np
from parallellinear.calculations.CalculationsManager import CalculationsManager
from parallellinear.calculations.NumpyLinear import NumpyLinear




class Matrix:

    CALCULATIONS_MANAGER=CalculationsManager(NumpyLinear.getLinearCalculator(), np.float32)

    def __init__(self, rows:int, data:np.ndarray, calcManager=None):
        self.rows = rows
        self.columns=int(len(data)/rows)
        self.data=data
        if calcManager==None:
            self.calcManager = Matrix.CALCULATIONS_MANAGER
        else:
            self.calcManager = calcManager

    @classmethod
    def setCalculationsManager(cls, calculationsManager:CalculationsManager):
        Matrix.CALCULATIONS_MANAGER = calculationsManager

    @classmethod
    def loadCustomFunctionToCalculationsManager(cls, function_name:str, func:str):
        Matrix.CALCULATIONS_MANAGER.getCalculator().loadCustomFunction(function_name, func)

    @classmethod
    def random(cls, rows:int, columns:int, calcManager=None, random_low=0, random_high=1):
        if calcManager==None:
            calcManager = Matrix.CALCULATIONS_MANAGER
        out = cls(rows=rows, data=np.random.rand(rows*columns).astype(calcManager.getPrecision()), calcManager=calcManager)
        if random_low != 0 or random_high != 1:
            out.scale(random_high-random_low)
            out.addScaler(random_low)
        return out 

    @classmethod
    def fromFlatListGivenRowNumber(cls, rows:int, data:list, calcManager=None,):
        if calcManager==None:
            calcManager = Matrix.CALCULATIONS_MANAGER
        if len(data) % rows != 0:
                raise ValueError("Matrix from list requires the following assertion to be true len(second parameter) % first parameter == 0")
        return cls(rows=rows, data=np.array(data, dtype=calcManager.getPrecision()), calcManager=calcManager)

    
    @classmethod
    def zeros(cls, rows:int, columns:int, calcManager=None):
        if calcManager==None:
            calcManager = Matrix.CALCULATIONS_MANAGER
        return cls(rows=rows, data=np.zeros(rows*columns, dtype=calcManager.getPrecision()), calcManager=calcManager)

    @classmethod
    def filledWithValue(cls, rows:int, columns:int, value, calcManager=None):
        if calcManager==None:
            calcManager = Matrix.CALCULATIONS_MANAGER
        return cls(rows=rows, data=np.empty(rows*columns, dtype=calcManager.getPrecision()).fill(value), calcManager=calcManager)


    def getNumberOfColumns(self):
        return self.columns

    def getNumberOfRows(self):
        return self.rows

    def setAtPos(self, x, y, val):
        self.data[x*self.columns+y] = val
    
    def getAtPos(self, x, y):
        return self.data[x*self.columns+y]

    def __str__(self):
        return str(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def getData(self):
        return self.data

    def exportToListRowNoAtLast(self):
        out=self.exportToList()
        out.append(self.rows)
        return out
    
    def exportToList(self):
        return self.data.tolist()


    def add(self, a, in_place = True) -> Any:
        if in_place:
            self.calcManager.getCalculator()._addInPlace(self.data, a.getData())
        else:
            return Matrix(self.rows, self.calcManager.getCalculator()._add(self.data, a.getData()), self.calcManager)

    def sub(self, a, in_place = True) -> Any:
        if in_place:
            self.calcManager.getCalculator()._subInPlace(self.data, a.getData())
        else:
            return Matrix(self.rows, self.calcManager.getCalculator()._sub(self, a), self.calcManager)

    def addScaler(self, scaler, in_place = True) -> Any:
        if in_place:
            self.calcManager.getCalculator()._addScalerInPlace(self.data, scaler)
        else:
            return Matrix(self.rows, self.calcManager.getCalculator()._addScaler(self.data, scaler), self.calcManager)

    def subScaler(self, scaler, in_place = True) -> Any:
        if in_place:
            self.calcManager.getCalculator()._subScalerInPlace(self.data, scaler)
        else:
            return Matrix(self.rows, self.calcManager.getCalculator()._subScaler(self.data, scaler), self.calcManager)

    def subScalerFrom(self, scaler, in_place = True) -> Any:
        if in_place:
            self.calcManager.getCalculator()._subScalerFromInPlace(self.data, scaler)
        else:
            return Matrix(self.rows, self.calcManager.getCalculator()._subScalerFrom(self.data, scaler), self.calcManager)

    def scale(self, scaler, in_place = True) -> Any:
        if in_place:
            self.calcManager.getCalculator()._scaleInPlace(self.data, scaler)
        else:
            return Matrix(self.rows, self.calcManager.getCalculator()._scale(self.data, scaler), self.calcManager)

    def descale(self, scaler, in_place = True) -> Any:
        if in_place:
            self.calcManager.getCalculator()._descale(self.data, scaler)
        else:
            return Matrix(self.rows, self.calcManager.getCalculator()._descale(self.data, scaler), self.calcManager)
    
    
    def __getitem__(self, indicies):
        return self.data[self.columns * indicies:self.columns*(indicies +1)]

    def transpose(self, in_place = True):
        
        bufferList = []
        for i in range(0, self.columns):
            for j in range(0, self.rows):
                bufferList.append(self.data[(j*self.columns)+i])
        

        if in_place:
            self.data = np.array(bufferList, dtype=self.calcManager.getPrecision())
            tmp=self.columns
            self.columns = self.rows
            self.rows = tmp
        else:
            return Matrix.fromFlatListGivenRowNumber(self.columns, bufferList)

    def multiply(self, a):
        return Matrix(self.rows, self.calcManager.getCalculator()._multiply(self.data, a.getData(), self.rows, self.columns, a.getNumberOfRows(), a.getNumberOfColumns()), self.calcManager)

    def applyCustomFunction(self, func_name, in_place = True):
        if in_place:
            self.calcManager.getCalculator()._applyCustomFunctionInPlace(self.data, func_name)
        else:
            return Matrix(self.rows, self.calcManager.getCalculator()._applyCustomFunction(self.data, func_name), self.calcManager)

    def sum(self):
        return self.calcManager.getCalculator()._sum(self.data)

    def elementWiseMultiply(self, a, in_place = True):
        if in_place:
            self.calcManager.getCalculator()._elementWiseMultiplyInPlace(self.data, a.getData())
        else:
            return Matrix(self.rows, self.calcManager.getCalculator()._elementWiseMultiply(self.data, a.getData()), self.calcManager)
            
        





