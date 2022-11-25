import time
import pickle

import numpy as np
import pyautogui
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import img_as_ubyte


def extract_color_features(image):
    # RGB histograms
    nbins = 16
    rh = np.histogram(image[:, :, 0].flatten(), nbins, density=True)
    gh = np.histogram(image[:, :, 1].flatten(), nbins, density=True)
    bh = np.histogram(image[:, :, 2].flatten(), nbins, density=True)

    # Mean and standard deviation of RGB channels
    r_mean = np.mean(image[:, :, 0])
    g_mean = np.mean(image[:, :, 1])
    b_mean = np.mean(image[:, :, 2])
    r_std = np.std(image[:, :, 0])
    g_std = np.std(image[:, :, 1])
    b_std = np.std(image[:, :, 2])

    return np.concatenate((rh[0], gh[0], bh[0], [r_mean, g_mean, b_mean, r_std, g_std, b_std]))


def extract_texture_features(image):
    # Convert to grayscale amd unsigned byte
    gray = img_as_ubyte(rgb2gray(image))

    # Extract texture-related features
    glcm = graycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    asm = graycoprops(glcm, 'ASM')

    return np.concatenate((contrast[0], dissimilarity[0], homogeneity[0], energy[0], correlation[0], asm[0]))


def extract_features(image):
    return np.concatenate((extract_color_features(image), extract_texture_features(image)))


def main():
    # Load the classifier
    classifier = pickle.load(open('model.pkl', 'rb'))

    while True:
        # Take a screenshot and resize it
        image = pyautogui.screenshot(region=(960, 540, 240, 135))
        image = np.array(image)
        image = resize(image, (135, 240, 3), anti_aliasing=True)

        # Extract features
        features = extract_features(image)

        # Predict the class
        prediction = classifier.predict([features])[0]

        # Show prediction and image
        plt.imshow(image)
        plt.title(f"Bioma: {prediction}")
        plt.show(block=False)
        plt.pause(10)
        plt.close()


if __name__ == '__main__':
    main()
