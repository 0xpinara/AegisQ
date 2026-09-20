"""Test suite root.

This file is load-bearing. Every test module imports its fixtures
absolutely (`from tests.conftest import random_circuit`), and without an
`__init__.py` the `tests` directory is only a *namespace portion*: the
import machinery records it and keeps scanning `sys.path`, so the first
directory anywhere on the path that contains a real `tests/__init__.py`
wins outright.

That is not hypothetical. On a machine with an unrelated project
installed in editable mode, `easy-install.pth` put its checkout on
`sys.path`, its `tests` package was a regular one, and this suite's
imports resolved into it -- another repository's conftest, in another
repository's directory. It surfaced as a collection error only because
the fixture names happened not to match. Had they matched, the tests
would have run green against a stranger's fixtures.

Making the package regular pins `tests` to this repository.
"""
