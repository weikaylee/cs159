"""Shared pytest configuration for the cs159-cloud-removal test suite.

This module is auto-loaded by pytest before any test collection. Its sole
responsibility is path setup: phase1_mirror_map/ is laid out as a flat
module directory (no __init__.py, matching the README's
`cd phase1_mirror_map && python train_mirror_map.py` invocation), so any
test that wants to `import icnn` / `import losses` / etc. needs that
directory on sys.path before the import line runs.

Putting the insert here means every test under tests/ — unit and
integration alike — gets the path setup for free. The existing inline
sys.path.insert at the top of tests/integration/test_train_mirror_map.py
becomes redundant but stays as-is (a duplicate insert is a no-op).
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
PHASE1 = os.path.abspath(os.path.join(REPO_ROOT, "..", "phase1_mirror_map"))
if PHASE1 not in sys.path:
    sys.path.insert(0, PHASE1)
