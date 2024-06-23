from AbstractDataLoader import AbstractDataLoader
from DicomLoader import DicomLoader
from NIFTILoader import NIFTILoader
from NRRDLoader import NRRDLoader


def getLoaderBasedOnFilename(filename) -> AbstractDataLoader:
    if filename.endswith('.dcm'):
        return DicomLoader()

    if filename.endswith('.nii.gz'):
        return NIFTILoader()

    if filename.endswith('.nrrd'):
        return NRRDLoader()
