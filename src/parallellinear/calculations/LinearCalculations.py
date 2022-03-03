from abc import ABC, abstractmethod

class LinearCalculations(ABC):


    def __init__(self):
        pass
    
    @classmethod
    @abstractmethod
    def getLinearCalculator(cls):
        pass

    @abstractmethod
    def loadCustomFunction(self, func_name, func):
        pass

    @abstractmethod
    def unloadCustomFunction(self, func_name):
        pass

    @abstractmethod
    def _addInPlace(self, a, b):
        pass

    @abstractmethod
    def _subInPlace(self, a, b):
        pass

    @abstractmethod
    def _elementWiseMultiplyInPlace(self, a, b):
        pass

    @abstractmethod
    def _addScalerInPlace(self, a, scaler):
        pass

    @abstractmethod
    def _subScalerInPlace(self, a, scaler):
        pass

    @abstractmethod
    def _subScalerFromInPlace(self, a, scaler):
        pass

    @abstractmethod
    def _scaleInPlace(self, a, scaler):
        pass

    @abstractmethod
    def _descaleInPlace(self, a, scaler):
        pass

    @abstractmethod
    def _add(self, a, b):
        pass

    @abstractmethod
    def _sub(self, a, b):
        pass

    @abstractmethod
    def _elementWiseMultiply(self, a, b):
        pass

    @abstractmethod
    def _addScaler(self, a, scaler):
        pass

    @abstractmethod
    def _subScaler(self, a, scaler):
        pass

    @abstractmethod
    def _subScalerFrom(self, a, scaler):
        pass

    @abstractmethod
    def _scale(self, a, scaler):
        pass

    @abstractmethod
    def _descale(self, a, scaler):
        pass

    @abstractmethod
    def _multiply(self, a, b, a_rows:int, a_cols:int, b_rows:int, b_cols:int):
        pass

    @abstractmethod
    def _applyCustomFunctionInPlace(self, a, func_name):
        pass

    @abstractmethod
    def _applyCustomFunction(self, a, func_name):
        pass

    @abstractmethod
    def _sum(self, a):
        pass