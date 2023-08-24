import numpy as np
from typing import Sequence
import json

def serialize(obj):
    # To check if the object is serializable
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        if isinstance(obj, Sequence):
            res = [serialize(x) for x in obj]
        # If we are dealing with a dict of object
        elif isinstance(obj, dict):
            res = {key : serialize(item) for key, item in obj.items()}
        # If we are dealing with a dict of object
        elif isinstance(obj, np.ndarray):
            res = obj.tolist()
        else:
            res =  jsonify(obj)
        return res

def jsonify(obj):
    # Find all class property names
    property_names = [
        p for p in dir(obj.__class__)
        # None is here in case "p" is not even an attribute
        if isinstance(getattr(obj.__class__, p, None), property)
    ]
    class_dict = {}
    for p_name in property_names:
        # property value
        p_value = getattr(obj, p_name)
        # Serialize value if necessary
        class_dict[p_name] = serialize(p_value)
    return class_dict