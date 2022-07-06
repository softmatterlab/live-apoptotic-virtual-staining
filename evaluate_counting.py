from __future__ import annotations
import glob
import os
import dlci
import json
import sys
import getopt
import numpy as np

from PIL import Image
from dlci import deeptrack as dt

import tensorflow as tf

_PATH_TO_DATASET = "../../data/"
_PATH_TO_MODELS = "models/loadstar counting"

# Grab passed arguments
opts, args = getopt.getopt(sys.argv[1:], "i:t:s:")

# Defaults
args = {
    "index": None,
    "set": None,
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

print("Loading calcein chem models...")
calcein_chem_model = dt.models.LodeSTAR(
    input_shape=(None, None, 1), loss="mae"
)
calcein_chem_model.build(input_shape=(None, None, None, 1))
calcein_chem_model.load_weights(
    os.path.join(_PATH_TO_MODELS, "model_calcein_chem.h5")
)

print("Loading calcein virtual models...")
calcein_virt_model = dt.models.LodeSTAR(
    input_shape=(None, None, 1), loss="mae"
)
calcein_virt_model.build(input_shape=(None, None, None, 1))
calcein_virt_model.load_weights(
    os.path.join(_PATH_TO_MODELS, "model_calcein_virt.h5")
)


print("Loading caspase chem models...")
caspase_chem_model = dt.models.LodeSTAR(
    input_shape=(None, None, 1), loss="mae"
)
caspase_chem_model.build(input_shape=(None, None, None, 1))
caspase_chem_model.load_weights(
    os.path.join(_PATH_TO_MODELS, "model_caspase_chem.h5")
)

print("Loading caspase virtual models...")
caspase_virt_model = dt.models.LodeSTAR(
    input_shape=(None, None, 1), loss="mae"
)
caspase_virt_model.build(input_shape=(None, None, None, 1))
caspase_virt_model.load_weights(
    os.path.join(_PATH_TO_MODELS, "model_caspase_virt.h5")
)

virt_models = {
    "calcein": calcein_virt_model,
    "caspase": caspase_virt_model,
}

parameters_virt_models = {
    "calcein": [2, 0.005, 8],
    "caspase": [1.2, 0.05, 4],
}

chem_models = {
    "calcein": calcein_chem_model,
    "caspase": caspase_chem_model,
}

parameters_chem_models = {
    "calcein": [1.0, 0.55, 6],
    "caspase": [1.4, 0.1, 6],
}


print("")
print("=" * 50, "START", "=" * 50)

filenames = glob.glob(
    os.path.join(_PATH_TO_DATASET, args["set"], "results", "*")
)
SITES = list(set([file.split("/")[-1] for file in filenames]))
print(SITES)
if args["index"]:
    SITES = list(filter(lambda fn: fn in args["index"], SITES))
print("Analyzing {} samples...".format(len(SITES)))

for site in SITES:

    for _type in ("calcein", "caspase"):
        try:
            
            annotations = []

            folder_path = os.path.join(
                _PATH_TO_DATASET, args["set"], "results", site, _type
            )
            files = glob.glob(os.path.join(folder_path, "*.tif"))

            print("Analyzing sample {}... saving to {}".format(site, folder_path))

            for idx, file in enumerate(files):
                print(file)

                print("----frame: " + str(idx))

                image = dt.LoadImage(path=file)()._value / 256
                image = image[::parameters_chem_models[_type][-1], ::parameters_chem_models[_type][-1]]

                pred, weights = chem_models[_type].predict(image[np.newaxis])
                positions = (
                    chem_models[_type].detect(
                        pred[0],
                        weights[0],
                        beta=1 - parameters_chem_models[_type][0],
                        alpha=parameters_chem_models[_type][0],
                        cutoff=parameters_chem_models[_type][1],
                        mode="constant",
                    )
                    * parameters_chem_models[_type][-1]
                )
                annotations.append(
                    dlci.get_annonations(
                        file=file,
                        positions=positions,
                    )
                )

            with open(
                os.path.join(
                    folder_path,
                    "detections.json",
                ),
                "w",
            ) as outfile:
                json.dump(annotations, outfile, cls=dlci.NumpyEncoder)

        except FileNotFoundError:
            pass
