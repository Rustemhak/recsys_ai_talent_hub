from http import HTTPStatus

import pytest
from starlette.testclient import TestClient

from service.api.app import create_app
from service.api.exceptions import UserNotAuthorized
from service.config import SECRET_PHRASE
from service.settings import get_config

config = get_config()
app = create_app(config)

client = TestClient(app)


def test_authorized_request():
    headers = {"Authorization": f"Bearer {SECRET_PHRASE}"}
    response = client.get("/health", headers=headers)
    assert response.status_code == HTTPStatus.OK


def test_unauthorized_request():
    headers = {"Authorization": "Bearer invalid_token"}
    with pytest.raises(UserNotAuthorized) as exc_info:
        client.get("/health", headers=headers)
    assert exc_info.value.status_code == HTTPStatus.UNAUTHORIZED
