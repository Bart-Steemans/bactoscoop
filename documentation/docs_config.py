"""Locate the package in either an in-repository or sibling documentation tree."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
override = os.environ.get('BACTOSCOOP_SOURCE')
PACKAGE = Path(override).expanduser().resolve() if override else (
    ROOT.parent if (ROOT.parent/'pyproject.toml').is_file() else ROOT.parent/'bactoscoop')
if not (PACKAGE/'bactoscoop/imagecollection.py').is_file():
    raise FileNotFoundError('Cannot find the package checkout; set BACTOSCOOP_SOURCE.')
