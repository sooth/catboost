import os  # noqa
import sys  # noqa

import pytest

try:
    import catboost_pytest_lib  # noqa
except ImportError:
    sys.path.append(os.path.join(os.environ['CMAKE_SOURCE_DIR'], 'catboost', 'pytest'))
    pytest_plugins = ["lib.common.pytest_plugin"]


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "fails_on_gpu(how): mark test that fails only on GPU"
    )


@pytest.fixture(params=['CPU'])
def task_type(request):
    return request.param


def _require_cuda():
    try:
        from catboost import utils
    except Exception as e:
        pytest.skip(f"catboost utils not available: {e}")
    try:
        devs = utils.get_gpu_device_count()
    except Exception as e:
        pytest.skip(f"CUDA not available in catboost: {e}")
    if devs < 1:
        pytest.skip("CUDA device not available")
    return devs


def _require_cupy():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        pytest.skip(f"cupy CUDA runtime unavailable: {e}")
    return cp


def _require_cudf():
    return pytest.importorskip("cudf")


@pytest.fixture
def require_cuda():
    return _require_cuda()


@pytest.fixture
def require_cupy():
    _require_cuda()
    return _require_cupy()


@pytest.fixture
def require_cudf():
    _require_cuda()
    _require_cupy()
    return _require_cudf()
