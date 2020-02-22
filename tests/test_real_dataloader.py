import pytest
from dataloader.dataloader import prepare_dataloader


@pytest.fixture
def dataloader():
    return True


def test_sanity(dataloader):
    dl = dataloader

    assert True
