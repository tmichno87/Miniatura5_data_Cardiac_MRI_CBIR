from AbstractDataLoader import AbstractDataLoader
import os
import numpy as np
import nibabel as nib


class NIFTILoader(AbstractDataLoader):
    def __init__(self):
        pass

    def loadFile(self, path, filename):
        filename = os.path.join(path, filename)
        img = nib.load(filename)
        return img.get_fdata()
