import cv2
import numpy as np
import math

class GammaCorrection:

    def __init__(self,ref_brightness = 0.55):
        self.ref_brightness = ref_brightness

    def find_brightness(self, image):
        """
        Find brightness of an image using image histogram
        :param image:
        :return:
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist(gray_image, [0], None, [256], [0, 256])
        pixels = np.sum(histogram)
        brightness = scale = len(histogram)

        for index in range(0, scale):
            ratio = histogram[index] / pixels
            brightness += ratio * (-scale + index)

        return 1 if brightness == 255 else brightness / scale

    def adjust_gamma(self, image, gamma=1.0):
        """
        Build a lookup table mapping the pixel values [0, 255]
        which corresponds to their adjusted gamma values
        Apply gamma correction using the lookup table
        :param image:
        :param gamma:
        :return:
        """
        table = np.array([((i / 255.0) ** gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def adaptive_gamma_correction(self, image):
        """
        Automatically adjust gamma value of an image.
        Adjust gamma to make average brightness (X) close to refrence brightness (R)
        gamma = log10(R)/log10(X) where X is average brightness
        :param image:
        :return:
        """
        brightness = self.find_brightness(image)[0]
        gamma = math.log10(self.ref_brightness) / math.log10(brightness)
        adjusted = self.adjust_gamma(image, gamma=gamma)
        return adjusted
