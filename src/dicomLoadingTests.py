import matplotlib.pyplot as plt
import DicomLoader

# Full path of the DICOM file is passed in base
base = r"/home/tm/Politechnika/miniatura/databases/Stanford Medicine/Ex Vivo Porcine Heart DT MRI Data/PIG_98979/Resolve_DTI/Resolve_Files/resolve_11seg_B1000_10x10x10_11/"
pass_dicom = "IM-0011-0021.dcm"  # file name is 1-12.dcm

# enter DICOM image name for pattern
# result is a list of 1 element
dcF = DicomLoader()
data = dcF.loadFile(base, pass_dicom)

# cmaps = [('Perceptually Uniform Sequential', [
#     'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
#     ('Sequential', [
#         'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
#     ('Sequential (2)', [
#         'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
#         'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
#         'hot', 'afmhot', 'gist_heat', 'copper']),
#     ('Diverging', [
#         'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
#         'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
#     ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
#     ('Qualitative', [
#         'Pastel1', 'Pastel2', 'Paired', 'Accent',
#         'Dark2', 'Set1', 'Set2', 'Set3',
#         'tab10', 'tab20', 'tab20b', 'tab20c']),
#     ('Miscellaneous', [
#         'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
#         'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
#         'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
#         'gist_ncar'])]

# for cmap_category, cmap_list in cmaps:
#     for cmap_name in cmap_list:
#         plt.imshow(ds.pixel_array, cmap=cmap_name)  # set the color map to bone
#         print(f"{cmap_category} colormaps, {cmap_name}")
#         plt.show()

plt.imshow(data, cmap=plt.cm.afmhot)  # set the color map to bone
plt.show()
#  gist_gray, gray, pink, hot, afmhot

# cmaps = [('Perceptually Uniform Sequential', [
#     'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
#     ('Sequential', [
#         'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
#     ('Sequential (2)', [
#         'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
#         'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
#         'hot', 'afmhot', 'gist_heat', 'copper']),
#     ('Diverging', [
#         'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
#         'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
#     ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
#     ('Qualitative', [
#         'Pastel1', 'Pastel2', 'Paired', 'Accent',
#         'Dark2', 'Set1', 'Set2', 'Set3',
#         'tab10', 'tab20', 'tab20b', 'tab20c']),
#     ('Miscellaneous', [
#         'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
#         'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
#         'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
#         'gist_ncar'])]


# def plot_color_gradients(cmap_category, cmap_list):
#     # Create figure and adjust figure height to number of colormaps
#     nrows = len(cmap_list)
#     figh = 0.35 + 0.15 + (nrows + (nrows-1)*0.1)*0.22
#     fig, axs = plt.subplots(nrows=nrows, figsize=(6.4, figh))
#     fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)

#     axs[0].set_title(f"{cmap_category} colormaps", fontsize=14)

#     for ax, cmap_name in zip(axs, cmap_list):
#         ax.imshow(ds.pixel_array, aspect='auto', cmap=cmap_name)
#         ax.text(-.01, .5, cmap_name, va='center', ha='right', fontsize=10,
#                 transform=ax.transAxes)

#     # Turn off *all* ticks & spines, not just the ones with colormaps.
#     for ax in axs:
#         ax.set_axis_off()


# for cmap_category, cmap_list in cmaps:
#     plot_color_gradients(cmap_category, cmap_list)

# plt.show()
