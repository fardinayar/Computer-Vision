import numpy as np
from matplotlib import pyplot as plt


class Schmid:
    def make_filter_bank(self):
        NF = 13
        SUP = 49
        F = np.zeros([SUP, SUP, NF])
        F[:,:, 0]= self.makefilter(SUP, 2, 1)
        F[:, :, 1] = self.makefilter(SUP, 4, 1)
        F[:, :, 2] = self.makefilter(SUP, 4, 2)
        F[:, :, 3] = self.makefilter(SUP, 6, 1)
        F[:, :, 4] = self.makefilter(SUP, 6, 2)
        F[:, :, 5] = self.makefilter(SUP, 6, 3)
        F[:, :, 6] = self.makefilter(SUP, 8, 1)
        F[:, :, 7] = self.makefilter(SUP, 8, 2)
        F[:, :, 8] = self.makefilter(SUP, 8, 3)
        F[:, :, 9] = self.makefilter(SUP, 10, 1)
        F[:, :, 10] = self.makefilter(SUP, 10, 2)
        F[:, :, 11] = self.makefilter(SUP, 10, 3)
        F[:, :, 12] = self.makefilter(SUP, 10, 4)
        return F

    def makefilter(self,sup,sigma,tau):
        hsup = (sup - 1) / 2
        x = [np.arange(-hsup, hsup + 1)]
        y = [np.arange(-hsup, hsup + 1)]
        [x, y] = np.meshgrid(x, y)
        r = (x * x + y * y) ** 0.5
        f = np.cos(r * (np.pi * tau / sigma)) * np.exp(-(r * r) / (2 * sigma * sigma));
        f = f - np.mean(f)
        f = f / sum(abs(f))
        return f

if __name__ == "__main__":
    schmid = Schmid().make_filter_bank()
    kernels = []
    for t in range(12):
        kernels.append(schmid[:, :, t])
        plt.subplot(4,3, t+1)
        plt.axis('off')
        plt.imshow(schmid[:, :, t], cmap='gray')
    plt.show()