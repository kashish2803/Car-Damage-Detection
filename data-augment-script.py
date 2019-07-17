import os
import random
from scipy import ndarray
import cv2
import matplotlib.pyplot as plt
import shutil

# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
}

folder_path = 'data/images_chem4/'
xml_path = 'data/annot_chem4/'

num_images_per_img = 10

for img,f in zip(os.listdir(folder_path), os.listdir(xml_path)):

    img_to_transform = sk.io.imread(os.path.join(folder_path, img))

    for i in range(num_images_per_img):

        key = random.choice(list(available_transformations))
        transformed_img = available_transformations[key](img_to_transform)
        new_file_path = f"data/images_chem4/augmented_{i}_{img}"
        sk.io.imsave(new_file_path, transformed_img)

        shutil.copy2(os.path.join(xml_path, f), os.path.join(xml_path, f"augmented_{i}_{f}"))

print(list(available_transformations))
print(images_path)
