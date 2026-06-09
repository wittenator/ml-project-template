"""Hydra registry for cluster profiles.

Adding a new cluster:
    1. Drop a file `scripts/conf/clusters/<name>.py` defining a
       `<Name>Profile = builds(ClusterProfile, ...)` literal.
    2. Add one `cluster_store(<Name>Profile, name="<name>")` line below.
    3. Use it: `./scripts/train.py cfg/cluster=<name>`.
"""

from conf.clusters.example import ExampleProfile
from conf.clusters.local import LocalProfile
from hydra_zen import store

# Sentinel guarding against re-registration: `configure_main` may run more than
# once per process (e.g. when several entry points share the same store).
# Mutating a list avoids rebinding a module global.
_registered: list[bool] = []


def register_clusters() -> None:
    if _registered:
        return
    cluster_store = store(group="cfg/cluster")
    cluster_store(LocalProfile, name="local")
    cluster_store(ExampleProfile, name="example")
    _registered.append(True)
