from http import HTTPStatus

import pytest
from httpx import AsyncClient

from service.api.app import create_app
from service.api.exceptions import UserNotAuthorized
from service.config import SECRET_PHRASE
from service.settings import get_config

config = get_config()
app = create_app(config)
HEADERS = {"Authorization": f"Bearer {SECRET_PHRASE}"}


@pytest.mark.asyncio
async def test_authorized_request():
    async with AsyncClient(app=app, base_url="http://localhost") as client:
        response = await client.get("/health", headers=HEADERS)
        assert response.status_code == HTTPStatus.OK


@pytest.mark.asyncio
async def test_unauthorized_request():
    async with AsyncClient(app=app, base_url="http://localhost") as client:
        with pytest.raises(UserNotAuthorized) as exc_info:
            await client.get("/health", headers=HEADERS)
        assert exc_info.value.status_code == HTTPStatus.UNAUTHORIZED
