import pytest
from dataloader.dataloader import prepare_dataloader


@pytest.fixture
def dataloader():
    return prepare_dataloader()


def test_sanity(dataloader):
    dl = dataloader(
    )

    assert True
