import os
import glob
import shutil

_validation_sites = [
    "01",
    "08",
    "09",
    "14",
    "21",
    "27",
    "31",
    "36",
    "42",
    "52",
]


def save_validation(dataset):

    for site in _validation_sites:

        DATASET_PATH = os.path.join(
            ".", "datasets", dataset, "training", "*_" + site + "_RAW*"
        )

        filenames = glob.glob(DATASET_PATH)

        for file in filenames:
            shutil.copy(file, file.replace(file.split("\\")[-2], "validation"))
            os.remove(file)


save_validation(dataset="Calcein")
