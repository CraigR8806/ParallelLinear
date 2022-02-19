from typing import Any
import numpy as np
import parallellinear.calculations.ParallelLinear as pl



class Matrix:

    def __init__(self, rows:int, data:np.ndarray):
        self.rows = rows
        self.columns=len(data)/rows
        self.data=data

    @classmethod
    def random(cls, rows:int, columns:int, random_low=0, random_high=1):
        out = cls(rows=rows, data=np.random.rand(rows*columns).astype(np.float32))
        if random_low != 0 or random_high != 1:
            out.scale(random_high-random_low)
            out.addScaler(random_low)
        return out 

    @classmethod
    def fromFlatListGivenRowNumber(cls, rows:int, data:list):
        if len(data) % rows != 0:
                raise ValueError("Matrix from list requires the following assertion to be true len(second parameter) % first parameter == 0")
        return cls(rows=rows, data=np.array(data).astype(np.float32))

    
    @classmethod
    def zeros(cls, rows, columns):
        return cls(rows=rows, data=np.zeros(rows*columns).astype(np.float32))

    @classmethod
    def filledWithValue(cls, rows:int, columns:int, value):
        return cls(rows=rows, data=np.ndarray(rows*columns).fill(value).astype(np.float32))


    def getNumberOfColumns(self):
        return self.columns

    def getNumberOfRows(self):
        return self.rows

    def setAtPos(self, x, y, val):
        self.data[int(x*self.columns+y)] = val
    
    def getAtPos(self, x, y):
        return self.data[int(x*self.columns+y)]

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
        return (lambda y:[x for a in y for x in (lambda z:z)(a)] if type(y) is list else [y])(self.data)


    def add(self, a, in_place = True) -> Any:
        if in_place:
            pl._addInPlace(self, a)
        else:
            return Matrix(self.rows, pl._add(self, a))

    def sub(self, a, in_place = True) -> Any:
        if in_place:
            pl._subInPlace(self, a)
        else:
            return Matrix(self.rows, pl._sub(self, a))

    def addScaler(self, a, in_place = True) -> Any:
        if in_place:
            pl._addScalerInPlace(self, a)
        else:
            return Matrix(self.rows, pl._addScaler(self, a))

    def subScaler(self, a, in_place = True) -> Any:
        if in_place:
            pl._subScalerInPlace(self, a)
        else:
            return Matrix(self.rows, pl._subScaler(self, a))

    def subScalerFrom(self, a, in_place = True) -> Any:
        if in_place:
            pl._subScalerFromInPlace(self, a)
        else:
            return Matrix(self.rows, pl._subScalerFrom(self, a))

    def scale(self, scaler, in_place = True) -> Any:
        if in_place:
            pl._scaleInPlace(self, scaler)
        else:
            return Matrix(self.rows, pl._scale(self, scaler))

    def descale(self, scaler, in_place = True) -> Any:
        if in_place:
            pl._descale(self, scaler)
        else:
            return Matrix(self.rows, pl._descale(self, scaler))
    
    
    def __getitem__(self, indicies):
        return self.data[int(self.columns * indicies):int(self.columns*(indicies +1))]

    def transpose(self, in_place = True):
        
        bufferList = []
        for i in range(0, int(self.columns)):
            for j in range(0, int(self.rows)):
                bufferList.append(self.data[int((j*self.columns)+i)])
        

        if in_place:
            self.data = np.array(bufferList).astype(np.float32)
            tmp=self.columns
            self.columns = self.rows
            self.rows = tmp
        else:
            return Matrix.fromFlatListGivenRowNumber(self.columns, bufferList)

    def multiply(self, a):
        return Matrix(self.rows, pl._multiply(self, a))

    def applyCustomFunction(self, func_name, in_place = True):
        if in_place:
            pl._applyCustomFunctionInPlace(self, func_name)
        else:
            return Matrix(self.rows, pl._applyCustomFunction(self, func_name))

    def elementWiseMultiply(self, a, in_place = True):
        if in_place:
            pl._elementWiseMultiplyInPlace(self, a)
        else:
            return Matrix(self.rows, pl._elementWiseMultiply(self, a))
            
        





