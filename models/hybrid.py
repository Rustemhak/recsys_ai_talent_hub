import json
from typing import List

from utils import load_recos, get_recos_from_dict

with open('recommenders/reco_paths.json') as jf:
    reco_paths = json.load(jf)

hybrid_recos = load_recos(reco_paths['hybrid_json'])


def get_recos_hybrid(user_id, k_recs=10) -> List:
    return get_recos_from_dict(user_id, recos=hybrid_recos, k_recs=k_recs)
