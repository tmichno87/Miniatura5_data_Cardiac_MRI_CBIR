from abc import ABC, abstractmethod
import numpy as np


class AbstractDataLoader(ABC):
    @abstractmethod
    def loadFile(self, path, filename):
        pass

    # convert data to have the same shape
    def convertToStack(self, data: np.ndarray):
        if data.ndim == 2:
            return data[:, :, np.newaxis, np.newaxis]
        if data.ndim == 3:
            return data[:, :, np.newaxis]
        if data.ndim > 3:
            return data
