import numpy as np
import cv2
import matplotlib.pyplot as plt


class DisparityMap():
    def __init__(self, disparity_range=64, half_block_size=7, size=1):
        self.disparity_range = disparity_range
        self.half_block_size = half_block_size
        self.size = size
    def __call__(self, left, right):
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        left_gray = cv2.equalizeHist(left_gray)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.equalizeHist(right_gray)
        left_gray = cv2.resize(left_gray, (0,0), fx=self.size, fy=self.size)
        plt.imshow(left_gray)
        plt.show()
        right_gray = cv2.resize(right_gray, (0,0), fx=self.size, fy=self.size)
        plt.imshow(right_gray)
        plt.show()
        self.map = np.zeros(left_gray.shape)
        for m in range(0, left_gray.shape[0]):
            min_r = max(0, m-self.half_block_size)
            max_r = min(left_gray.shape[0],m+self.half_block_size)
            for n in range(0, left_gray.shape[1]):
                min_c = max(0, n-self.half_block_size)
                max_c = min(left_gray.shape[1], n+self.half_block_size)

                min_d_y = 0
                max_d_y = min(self.disparity_range, left_gray.shape[1]-max_c)

                cost = np.inf
                for d_y in range(min_d_y, max_d_y):

                    disparity_cost = np.sum(np.square(left_gray[min_r:max_r, min_c:max_c] - right_gray[min_r:max_r, min_c + d_y:max_c + d_y]))
                    if disparity_cost < cost:
                        cost = disparity_cost
                        self.map[m,n] = d_y
                        print(m)

        return self.map


if __name__ == '__main__':
    left = cv2.imread('im6.png')
    right = cv2.imread('im2.png')
    disparitymap = DisparityMap(size=1)
    map = disparitymap(left, right)
    plt.imshow(map, cmap='gray')
    plt.show()