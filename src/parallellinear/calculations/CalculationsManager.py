from parallellinear.calculations.LinearCalculations import LinearCalculations
import numpy as np


class CalculationsManager:

    def __init__(self, calculator:LinearCalculations, precision:np.dtype):
        self.calculator = calculator
        self.precision = precision

    def getPrecision(self):
        return self.precision

    def getCalculator(self):
        return self.calculator