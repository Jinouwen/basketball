import numpy as np
import cv2


# 用法： my_hog = Hog((image_height, image_width), block_size in cell, block_stride in cell, cell_size in pixel,
# full-angle, nbin)
# 例如：cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9) <=> Hog((128, 64), (2, 2), (1, 1), (8, 8))
# v = my_hog.compute(img) 即得到HOG向量

class Hog:

    def __init__(self, window_size, block_size, block_stride, cell_size, full_angle=180, nbin = 9):
        assert len(window_size) == 2
        assert len(block_size) == 2
        assert len(block_stride) == 2
        assert len(cell_size) == 2
        assert window_size[0] % cell_size[0] == 0
        assert window_size[1] % cell_size[1] == 0
        cell_number = (window_size[0] / cell_size[0], window_size[1] / cell_size[1])
        assert cell_number[0] >= block_size[0] >= 0
        assert cell_number[1] >= block_size[1] >= 0
        assert block_stride[0] > 0 and (cell_number[0] - block_size[0]) % block_stride[0] == 0
        assert block_stride[1] > 0 and (cell_number[1] - block_size[1]) % block_stride[1] == 0
        self.window_size = window_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbin = nbin
        self.full_angle = full_angle

    @property
    def cell_number(self):
        return int(self.window_size[0] / self.cell_size[0]), int(self.window_size[1] / self.cell_size[1])

    @property
    def block_number(self):
        return int((self.cell_number[0] - self.block_size[0]) / self.block_stride[0] + 1), int(
            (self.cell_number[1] - self.block_size[1]) / self.block_stride[1] + 1)

    @property
    def bin_width(self):
        return self.full_angle / self.nbin

    @property
    def bin_value(self, index: int):
        assert 0 <= index < self.nbin
        return self.bin_width * index

    def compute_gradient(self, img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        magnitude, orientation = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        index = np.argmax(magnitude, axis=-1)
        magnitude = from_index(magnitude, index)
        orientation = from_index(orientation, index)
        return magnitude, orientation

    def compute_cell_histogram(self, img):
        hists = np.zeros((*self.cell_number, self.nbin))
        magnitude, orientation = self.compute_gradient(img)
        grads = np.zeros((*self.window_size, 2))
        bins = np.zeros((*self.window_size, 2), dtype=np.int)
        for row in range(self.cell_number[0]):
            for col in range(self.cell_number[1]):
                angle = (orientation[row * 8:(row + 1) * 8, col * 8:(col + 1) * 8] % self.full_angle)
                weight = magnitude[row * 8:(row + 1) * 8, col * 8:(col + 1) * 8]
                for y in range(angle.shape[0]):
                    for x in range(angle.shape[1]):
                        if angle[y][x] < self.bin_width / 2 or angle[y][x] > self.full_angle - self.bin_width / 2:
                            left_bin = self.nbin - 1
                            right_bin = 0
                            if angle[y][x] < self.bin_width / 2:
                                left_weight = (self.bin_width / 2 + angle[y][x]) / self.bin_width
                                right_weight = (self.bin_width / 2 - angle[y][x]) / self.bin_width
                            else:
                                left_weight = (self.bin_width / 2 - (self.full_angle - angle[y][x])) / self.bin_width
                                right_weight = (self.bin_width / 2 + (self.full_angle - angle[y][x])) / self.bin_width
                        else:
                            # if row == 0 and col == 0 and y == 1 and x == 2:
                            #     print("here")
                            #     print(angle[y][x])
                            pos = angle[y][x] - self.bin_width / 2
                            left_bin = int(pos / self.bin_width)
                            right_bin = left_bin + 1
                            left_weight = (pos - self.bin_width * left_bin) / self.bin_width
                            right_weight = (self.bin_width * right_bin - pos) / self.bin_width
                        left_weight *= weight[y][x]
                        right_weight *= weight[y][x]
                        hists[row][col][left_bin] += left_weight
                        hists[row][col][right_bin] += right_weight
                        grads[int(row * self.cell_size[0]) + y][int(col * self.cell_size[1]) + x][0] = left_weight
                        grads[int(row * self.cell_size[0]) + y][int(col * self.cell_size[1]) + x][1] = right_weight
                        bins[int(row * self.cell_size[0]) + y][int(col * self.cell_size[1]) + x][0] = left_bin
                        bins[int(row * self.cell_size[0]) + y][int(col * self.cell_size[1]) + x][1] = right_bin
        return hists, grads, bins

    def compute(self, img):
        v = np.array([])
        hists, grads, bins = self.compute_cell_histogram(img)
        print(self.block_number)
        for row in range(self.block_number[0]):
            for col in range(self.block_number[1]):
                temp = np.array([])
                for i in range(self.block_size[0]):
                    for j in range(self.block_size[1]):
                        temp = np.concatenate((temp, hists[i + row][j + col]))
                distance = np.sqrt(np.sum(temp ** 2) + 0.1 * len(temp))
                temp /= distance
                temp[temp > 0.2] = 0.2
                distance = np.sqrt(np.sum(temp ** 2) + 0.001)
                temp /= distance
                v = np.concatenate((v, temp))
        return v


def from_index(ndarray, index):
    if ndarray is not None and index is not None:
        flat_idx = np.arange(ndarray.size, step=ndarray.shape[-1]) + index.ravel()
        return ndarray.ravel()[flat_idx].reshape(*ndarray.shape[:-1])
    else:
        return None
