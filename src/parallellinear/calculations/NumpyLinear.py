from parallellinear.calculations.LinearCalculations import LinearCalculations
import numpy as np


class NumpyLinear(LinearCalculations):



    def __init__(self):
        pass

    @classmethod
    def getLinearCalculator(cls):
        return cls()
    
    def loadCustomFunction(self, func_name, func):
        pass

    
    def unloadCustomFunction(self, func_name):
        pass

    
    def _addInPlace(self, a, b):
        a += b

    
    def _subInPlace(self, a, b):
        a -= b

    
    def _elementWiseMultiplyInPlace(self, a, b):
        a *= b

    
    def _addScalerInPlace(self, a, scaler):
        data += scaler

    
    def _subScalerInPlace(self, a, scaler):
        a -= scaler

    
    def _subScalerFromInPlace(self, a, scaler):
        a = scaler - a

    
    def _scaleInPlace(self, a, scaler):
        a *= scaler

    
    def _descaleInPlace(self, a, scaler):
        a /= scaler

    
    def _add(self, a, b):
        return a + b

    
    def _sub(self, a, b):
        return a - b

    
    def _elementWiseMultiply(self, a, b):
        return a * b

    
    def _addScaler(self, a, scaler):
        return a + scaler

    
    def _subScaler(self, a, scaler):
        return a - scaler

    
    def _subScalerFrom(self, a, scaler):
        return scaler - a

    
    def _scale(self, a, scaler):
        return a * scaler

    
    def _descale(self, a, scaler):
        return a / scaler

    
    def _multiply(self, a, b, a_rows:int, a_cols:int, b_rows:int, b_cols:int):
        amd=np.array([a[i:i+a_cols] for i in range(0, len(a), a_cols)], dtype=a.dtype)
        bmd=np.array([b[i:i+b_cols] for i in range(0, len(b), b_cols)], dtype=a.dtype)
        return np.array(amd.dot(bmd).flat, dtype=a.dtype)

    
    def _applyCustomFunctionInPlace(self, a, func_name):
        pass

    
    def _applyCustomFunction(self, a, func_name):
        pass