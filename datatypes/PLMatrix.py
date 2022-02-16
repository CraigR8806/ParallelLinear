from typing import Any
import numpy as np
import plinear.calculations.ParallelLinear as pl



class Matrix:

    rows=0
    columns=0
    data=None

    def __init__(self, rows, colsOrVals, **kwargs):
        if type(rows) != int:
            raise TypeError("Matrix constructor requires the first param to be an int representing the number of rows")
        self.rows=rows
        if type(colsOrVals) == int:
            self.columns=colsOrVals
            length=self.rows * self.columns
            
            if kwargs.get('random', False):
                self.data = np.random.rand(length).astype(np.float32)
            else:
                self.data = np.zeros(length)

        elif type(colsOrVals) == list:
            if len(colsOrVals) % rows != 0:
                raise ValueError("Matrix constructor, when used with second parameter as list, requires the following assertion to be true len(second parameter) % first parameter == 0")
            self.columns=len(colsOrVals)/rows
            self.data = np.array(colsOrVals).astype(np.float32)
        elif type(colsOrVals) == np.ndarray:
            if len(colsOrVals) % rows != 0:
                raise ValueError("Matrix constructor, when used with second parameter as list, requires the following assertion to be true len(second parameter) % first parameter == 0")
            self.columns=len(colsOrVals)/rows
            self.data = colsOrVals


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
            return Matrix(self.getNumberOfRows(), pl._add(self, a))

    def sub(self, a, in_place = True) -> Any:
        if in_place:
            pl._subInPlace(self, a)
        else:
            return Matrix(self.getNumberOfRows(), pl._sub(self, a))

    def addScaler(self, a, in_place = True) -> Any:
        if in_place:
            pl._addScalerInPlace(self, a)
        else:
            return Matrix(self.getNumberOfRows(), pl._addScaler(self, a))

    def subScaler(self, a, in_place = True) -> Any:
        if in_place:
            pl._subScalerInPlace(self, a)
        else:
            return Matrix(self.getNumberOfRows(), pl._subScaler(self, a))

    def scale(self, scaler, in_place = True) -> Any:
        if in_place:
            pl._scaleInPlace(self, scaler)
        else:
            return Matrix(self.getNumberOfRows(), pl._scale(self, scaler))

    def descale(self, scaler, in_place = True) -> Any:
        if in_place:
            pl._descale(self, scaler)
        else:
            return Matrix(self.getNumberOfRows(), pl._descale(self, scaler))
    
    
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
            return Matrix(int(self.columns), bufferList)

    def multiply(self, a):
        return Matrix(self.rows, pl._multiply(self, a))

    def applyCustomFunction(self, func_name, in_place = True):
        if in_place:
            pl._applyCustomFunctionInPlace(self, func_name)
        else:
            return Matrix(pl._applyCustomFunction(self, func_name))
            
        





