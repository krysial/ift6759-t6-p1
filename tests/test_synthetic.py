import pytest

from utils.synthetic import SyntheticMNISTGenerator, Options


opts = Options(100, 12)


@pytest.fixture
def setup_generator():
    def _setup_generator():
        return SyntheticMNISTGenerator(opts)

    return _setup_generator


def test_answer(setup_generator):
    generator = setup_generator()
    assert generator is not None
