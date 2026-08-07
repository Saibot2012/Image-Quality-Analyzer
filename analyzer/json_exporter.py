import json
import os
import numpy as np


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(
        f"Object of type {type(obj)} is not JSON serializable."
    )

def save_json_report(
        report,
        filename,
):
    os.makedirs(
        "JSON",
        exist_ok=True
    )
    name = os.path.splitext(filename)[0]

    path = os.path.join(
        "JSON",
        name + ".json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=4,
            default = convert_numpy
        )
    return path
