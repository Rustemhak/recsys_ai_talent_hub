import os
import pickle
import pandas as pd
from rectools import Columns
from rectools.tools import UserToItemAnnRecommender
from models.load_models import load_popular

lightfm_ann = None
MODEL_PATH = "models/ANN_LightFM.pkl"
if os.path.exists(MODEL_PATH):
    lightfm_ann = pickle.load(open(MODEL_PATH, 'rb'))


all_recos = pd.DataFrame([])
all_users = []
RECOS_PATH = "recommenders/offline/LightFM.csv"
if os.path.exists(RECOS_PATH):
    all_recos = pd.read_csv(RECOS_PATH)
    all_users = all_recos[Columns.User].unique()
model_popular = load_popular()


popular_recos = pd.read_csv(os.path.join('kion_train', "popular_50.csv")).item_id.tolist()


def get_offline_recos_lightfm(user_id):
    if user_id in all_users:
        user_recos = all_recos[all_recos[Columns.User] == user_id][Columns.Item]
        return user_recos.tolist()
    return popular_recos[:10]

def get_recos_lightfm_ann(user_id, k_recs=10):
    if user_id in lightfm_ann.user_id_map.external_ids:
        return lightfm_ann.get_item_list_for_user(user_id, top_n=k_recs).tolist()
    else:
        return model_popular.predict([[user_id]])
