import os
import json
from typing import Dict, List

from load_models import load_popular
model_popular = load_popular()

def load_recos(path):
    if os.path.exists(path):
        with open(path) as jf:
            recos = json.load(jf)
        return recos
    return {}


def get_recos_from_dict(user_id, recos:Dict[str, List[int]], k_recs=10):
    user_id = str(user_id)
    if user_id in recos:
        return recos[user_id][:k_recs]
    return model_popular.predict([[user_id]])
