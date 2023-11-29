from implicit.nearest_neighbours import CosineRecommender, TFIDFRecommender


class UserKnn_model_conf():
    model = TFIDFRecommender()
    weight_path = 'data/weights/userknn_tfidf_50.dill'
    save_reco_df_path = 'data/offline_reco/userknn_TFIDF.csv'
    N_recs = 10
    online = False
    n_folds = 3
    unit = "D"
    n_units = 4
    dataset_path = 'data/kion_train'


class Popular_model_conf:
    save_reco_df_path = '../data/offline_reco/popular.csv'
    dataset_path = 'data/kion_train'
    N_recs = 10
