from typing import List

import dill
from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel

from models.nn_models import get_recos_AE, get_recos_DSSM, get_recos_multi_VAE
from service.api.exceptions import ModelNotFoundError, UserNotFoundError
from service.log import app_logger
from service.ml_models import Random

# Init models

model_knn = None
with open("data/weights/userknn_tfidf_50.dill", "rb") as f:
    model_knn = dill.load(f)

model_popular = None
with open("data/weights/popular.dill", "rb") as f:
    model_popular = dill.load(f)

model_lightfm = None
with open("data/weights/lightfm.dill", "rb") as f:
    model_lightfm = dill.load(f)

random_model = Random()


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()


@router.get(
    path="/health",
    tags=["Health"],
    summary="Health Check",
    response_description="Check if the API is healthy",
    responses={
        200: {"description": "API is healthy"},
    },
)
async def health() -> str:
    return "API is healthy"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    response_description="Get recommendations for a user based on a model",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {"user_id": 111, "items": [334, 343, 324, 656, 6785, 345, 1242, 34534, 234, 23]}
                }
            },
        },
        404: {
            "description": "Model not found",
            "content": {"application/json": {"example": {"detail": "Model not found"}}},
        },
    },
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")
    k_recs = request.app.state.k_recs

    if model_name == "random":
        reco = list(range(k_recs))
    elif model_name == "popular":
        reco = model_popular.predict([[user_id]])
    elif model_name == "knn":
        reco = model_knn.predict_online([[user_id]])
    elif model_name == "lightfm":
        reco = model_lightfm.predict([[user_id]])
    elif model_name == "dssm":
        reco = get_recos_DSSM(user_id, k_recs=k_recs)
    elif model_name == "ae":
        reco = get_recos_AE(user_id, k_recs=k_recs)
    elif model_name == "multi_vae":
        reco = get_recos_multi_VAE(user_id, k_recs=k_recs)
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")
    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
