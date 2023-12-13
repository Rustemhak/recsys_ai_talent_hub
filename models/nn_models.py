import os
import json
from typing import Dict, List
from models.load_models import load_popular

with open('recommenders/reco_paths.json') as jf:
    reco_paths = json.load(jf)
model_popular = load_popular()


def load_recos(path):
    if os.path.exists(path):
        with open(path) as jf:
            recos = json.load(jf)
        return recos
    return {}


AE_recos = load_recos(reco_paths['ae_json'])
multi_VAE_recos = load_recos(reco_paths['multi_vae_json'])
DSSM_recos = load_recos(reco_paths['dssm_json'])


def get_recos_from_dict(user_id, recos: Dict[str, List[int]],
                        k_recs=10) -> List:
    user_id = str(user_id)
    if user_id in recos:
        return recos[user_id][:k_recs]
    return model_popular.predict([[user_id]])


def get_recos_AE(user_id, k_recs=10) -> List:
    return get_recos_from_dict(user_id, recos=AE_recos, k_recs=k_recs)


def get_recos_multi_VAE(user_id, k_recs=10) -> List:
    return [int(r) for r in
            get_recos_from_dict(user_id, recos=multi_VAE_recos, k_recs=k_recs)]


def get_recos_DSSM(user_id, k_recs=10) -> List:
    return get_recos_from_dict(user_id, recos=DSSM_recos, k_recs=k_recs)
