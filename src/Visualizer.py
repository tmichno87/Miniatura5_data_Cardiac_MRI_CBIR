import matplotlib.pyplot as plt


def ShowMRIData(data, stackNo=0, timeNo=0):
    plt.imshow(data[:, :, stackNo, timeNo], cmap=plt.cm.afmhot)
    plt.show()
