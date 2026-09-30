"""Pin `docs/metrics.md` to the metrics the trainer actually emits.

The doc is a reference, not a source of truth, so it can drift silently: adding
a diagnostic to the losses never has to touch it. This runs a real update and
checks every emitted name is documented, which turns that drift into a failing
test instead of a reader's wrong assumption.

Only trainer-emitted namespaces are checked. The collector-side names
(`episode/*`, `evaluation/<scope>/*`) come from callbacks that need a live
environment and are documented by hand.
"""

from importlib import util as importlib_util
from pathlib import Path
import re

import pytest

DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "metrics.md"
INVENTORY_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmarks"
    / "metric_inventory.py"
)

# Names the doc intentionally describes by pattern rather than listing verbatim.
DOCUMENTED_PREFIXES = ("evaluation/", "episode/")
DOCUMENTED_SUFFIXES = ("lr_{i}",)

# A documented `metrics/name_{i}` row stands in for every concrete expansion of
# that family: listing all three action dimensions separately would triple the
# table for no added information.
_FAMILY_SUFFIX = re.compile(r"_\d+$")


def _load_inventory_module():
    """Import the generator script by path; `scripts/` is not a package."""
    spec = importlib_util.spec_from_file_location("metric_inventory", INVENTORY_PATH)
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _documented_names(doc: str) -> set[str]:
    """Every metric name the doc mentions verbatim."""
    return set(re.findall(r"`([a-z_]+/[a-z_0-9{}_]+)`", doc))


def _is_documented(name: str, documented: set[str]) -> bool:
    """Whether the doc covers an emitted name, directly or via its family row."""
    if name in documented:
        return True
    # `metrics/alpha_0` is covered by a documented `metrics/alpha_{i}` row.
    return _FAMILY_SUFFIX.sub("_{i}", name) in documented


@pytest.fixture(scope="module")
def emitted_names() -> set[str]:
    return set(_load_inventory_module().emitted_metric_names())


@pytest.fixture(scope="module")
def documented_names() -> set[str]:
    return _documented_names(DOC_PATH.read_text())


def test_the_generator_finds_metrics(emitted_names: set[str]):
    """Guards the fixture: an empty capture would make the checks below vacuous."""
    assert emitted_names
    assert {"loss/total", "metrics/approx_kl", "rollout/returns"} <= emitted_names


def test_every_emitted_metric_is_documented(
    emitted_names: set[str], documented_names: set[str]
):
    undocumented = {
        name
        for name in emitted_names
        if not _is_documented(name, documented_names)
        and not name.startswith(DOCUMENTED_PREFIXES)
        and not name.endswith(DOCUMENTED_SUFFIXES)
    }

    assert not undocumented, (
        f"{len(undocumented)} metric(s) emitted but missing from {DOC_PATH.name}: "
        f"{sorted(undocumented)}. Document them there, then regenerate with "
        "`uv run --extra cpu python scripts/benchmarks/metric_inventory.py`."
    )


def test_the_documented_generator_is_reachable():
    """The doc tells readers to regenerate; the script it names has to exist."""
    assert INVENTORY_PATH.is_file()
    assert "scripts/benchmarks/metric_inventory.py" in DOC_PATH.read_text()