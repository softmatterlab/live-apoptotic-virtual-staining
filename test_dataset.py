import dlci
import matplotlib.pyplot as plt
import numpy as np
from dlci import deeptrack as dt


def get_image(image: dt.Image) -> dt.Image:
    images = image[0]

    return images


def get_mask(image: dt.Image) -> dt.Image:
    return image[1]


PATH = r"C:/GU/Live-dead-staining/datasets"

augmentation_list = {
    "FlipLR": {},
    "FlipUD": {},
    "FlipDiagonal": {},
    "Affine": {
        "rotate": lambda: np.random.rand() * 2 * np.pi,
        "shear": lambda: np.random.rand() * 0.3 - 0.15,
        "scale": {
            "x": np.random.rand() * 0.3 + 0.85,
            "y": np.random.rand() * 0.3 + 0.85,
        },
    },
}


dataset = dlci.DataLoader(
    path_to_dataset=PATH, dataset="Caspase", augmentation=augmentation_list
)

test = np.zeros((4, 4, 7))

NUMBER_OF_IMAGES = 8
for image_index in range(NUMBER_OF_IMAGES):
    image_tuple = dataset.update(validation=False).resolve()
    image = get_image(image_tuple)
    mask = get_mask(image_tuple)
    print(np.shape(mask))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, 0])
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(mask[:, :, 0])
    plt.colorbar()
    plt.show()
