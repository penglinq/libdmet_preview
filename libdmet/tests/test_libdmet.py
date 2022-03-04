"""
Unit and regression test for the libdmet package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import libdmet


def test_libdmet_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "libdmet" in sys.modules
