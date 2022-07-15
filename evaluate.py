import glob
import os
import dlci
import json
import sys
import getopt
import numpy as np

from PIL import Image
from dlci import deeptrack as dt

_PATH_TO_DATASET = "data"
_PATH_TO_MODELS = "/virtual staining/models"

# Grab passed arguments
opts, args = getopt.getopt(sys.argv[1:], "i:t:s:")

# Defaults
args = {
    "index": None,
    "set": "1",
    "save": True,
}

for opt, arg in opts:
    if opt == "-i":
        args["index"] = [i for i in arg.split("-")]
    elif opt == "-t":
        args["set"] = arg
    elif opt == "-s":
        args["save"] = arg == "True"

print("Loading models...")

calcein = dlci.load_model(
    glob.glob(os.path.join(_PATH_TO_MODELS, "model_calcein", "*"))[0]
)
calcein.compile(loss="mae")
caspase = dlci.load_model(
    glob.glob(os.path.join(_PATH_TO_MODELS, "model_caspase", "*"))[0]
)
caspase.compile(loss="mae")

print("")
print("=" * 50, "START", "=" * 50)

network = (
    dt.Lambda(
        lambda: lambda image: [
            calcein.predict(np.expand_dims(image, axis=0)),
            caspase.predict(np.expand_dims(image, axis=0)),
        ]
    )
    + dt.Multiply(0.5)
    + dt.Add(0.5)
)

filenames = glob.glob(
    os.path.join(_PATH_TO_DATASET, "set " + args["set"], "*ch00*.tif")
)
SITES = list(set([file[-19:-17] for file in filenames]))
if args["index"]:
    SITES = list(filter(lambda fn: fn in args["index"], SITES))

print("Analyzing {} samples...".format(len(SITES)))

for site in SITES:
    _filenames = list(filter(lambda fn: fn[-19:-17] == site, filenames))

    for _type in ("calcein", "caspase"):
        folder_path = os.path.join(
            _PATH_TO_DATASET, "set " + args["set"], "results", site, _type
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    print("Analyzing sample {}... saving to {}".format(site, folder_path))

    for idx, file in enumerate(_filenames):

        print(file)

        print("----frame: " + str(idx))

        loader = (
            dt.LoadImage(path=file)
            + dt.PadToMultiplesOf(multiple=(32, 32, None))
            + dt.NormalizeMinMax(min=-1, max=1)
        )

        staining = loader + network

        image_calcein, image_caspase = staining.update().resolve()

        image_calcein = np.array(image_calcein)
        image_caspase = np.array(image_caspase)

        if args["save"]:
            im = Image.fromarray(
                (255 * image_calcein[0, ..., 0]).astype(np.uint8)
            )
            im.save(
                os.path.join(
                    _PATH_TO_DATASET,
                    "set " + args["set"],
                    "results",
                    site,
                    "calcein",
                    file.split("\\")[2].replace("ch00", "ch01"),
                )
            )

            im = Image.fromarray(
                (255 * image_caspase[0, ..., 0]).astype(np.uint8)
            )
            im.save(
                os.path.join(
                    _PATH_TO_DATASET,
                    "set " + args["set"],
                    "results",
                    site,
                    "caspase",
                    file.split("\\")[2].replace("ch00", "ch01"),
                )
            )
