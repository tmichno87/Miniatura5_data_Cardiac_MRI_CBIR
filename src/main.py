from AbstractDataLoader import AbstractDataLoader
from LoaderFactory import *
from Visualizer import *


# Full path of the DICOM file is passed in base
# base = r"/home/tm/Politechnika/miniatura/databases/Stanford Medicine/Ex Vivo Porcine Heart DT MRI Data/PIG_98979/Resolve_DTI/Resolve_Files/resolve_11seg_B1000_10x10x10_11/"
# filename = "IM-0011-0021.dcm"

# # Full path of the NIFTI file is passed in base
# base = r"/home/tm/Politechnika/miniatura/databases/ACDC_2017/training/patient001/"
# filename = "patient001_4d.nii.gz"


base = r"/home/tm/Politechnika/miniatura/databases/typyMRI/MnM2/mnm2/MnM2/dataset/001/"
filename = "001_LA_CINE.nii.gz"

dataLoader = getLoaderBasedOnFilename(filename)
data = dataLoader.loadFile(base, filename)
# ShowMRIData(data[:, :, 1, 0])

print(data.shape)
stacks = dataLoader.convertToStack(data)
ShowMRIData(stacks)
print(stacks.shape)
