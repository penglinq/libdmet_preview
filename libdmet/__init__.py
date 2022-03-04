"""A periodic density matrix embedding theory library for correlated materials."""

# Add imports here
from .libdmet import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

__doc__ = \
"""
libDMET   version %s
A periodic density matrix embedding theory implementation for lattice model and realistic solid.
""" % (__version__)
