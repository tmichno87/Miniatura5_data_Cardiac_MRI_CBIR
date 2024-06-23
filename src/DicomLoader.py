from AbstractDataLoader import AbstractDataLoader
import pydicom
import pydicom.data


class DicomLoader(AbstractDataLoader):
    def __init__(self):
        pass

    def loadFile(self, path, filename):
        filename = pydicom.data.data_manager.get_files(path, filename)[0]
        ds = pydicom.dcmread(filename)
        return ds.pixel_array
