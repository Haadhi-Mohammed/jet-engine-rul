# tests for the Streamlit dashboard
# runs the real dashboard script against the real API — in memory, no network:
# the dashboard's requests.get() calls are routed into FastAPI's TestClient.

import re
from urllib.parse import urlparse

import pytest
import requests
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import api.main as api

APP = 'dashboard/app.py'
api_client = TestClient(api.app)


class FakeResponse:
    # just enough of requests.Response for the dashboard
    def __init__(self, response):
        self._response = response
        self.status_code = response.status_code

    def json(self):
        return self._response.json()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)


@pytest.fixture(autouse=True)
def dashboard_env(monkeypatch):
    monkeypatch.setenv('RUL_API_URL', 'http://fake-api')
    monkeypatch.setenv('RUL_API_WAKE_BUDGET', '0')     # don't spend 3 minutes retrying in tests
    st.cache_data.clear()                               # each test starts with a cold cache


@pytest.fixture
def api_up(monkeypatch):
    monkeypatch.setattr(requests, 'get',
                        lambda url, timeout=None: FakeResponse(api_client.get(urlparse(url).path)))


def text_of(at) -> str:
    # all markdown on the page, HTML tags stripped
    return ' '.join(re.sub('<[^>]+>', ' ', m.value) for m in at.markdown)


def test_fleet_view_shows_the_api_numbers(api_up):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception and not at.error

    fleet = api_client.get('/fleet').json()
    page = text_of(at)
    for key in ('red_count', 'amber_count', 'green_count'):
        assert str(fleet[key]) in page
    assert 'Actual RUL' in at.dataframe[0].value.columns


def test_drill_down_explains_the_selected_engine(api_up):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value('Engine Drill-down').run()
    [selector] = [s for s in at.selectbox if s.label == 'Select Engine']
    selector.set_value(34).run()
    assert not at.exception and not at.error

    explain = api_client.get('/engines/34/explain').json()
    page = text_of(at)
    assert f"{explain['predicted_rul']:.1f} cycles" in page
    assert 'Immediate inspection recommended' in page            # engine 34 is RED
    assert explain['shap_values'][0]['sensor'] in page           # top sensor is named


def test_api_down_shows_a_message_instead_of_crashing(monkeypatch):
    def unreachable(url, timeout=None):
        raise requests.ConnectionError('connection refused')
    monkeypatch.setattr(requests, 'get', unreachable)

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert 'API request failed' in at.error[0].value
