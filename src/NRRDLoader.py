from AbstractDataLoader import AbstractDataLoader
import os
import numpy as np
import nrrd


class NRRDLoader(AbstractDataLoader):
    def __init__(self):
        pass

    def loadFile(self, path, filename):
        filename = os.path.join(path, filename)
        readdata, header = nrrd.read(filename)
        return readdata
