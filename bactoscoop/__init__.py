# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:49:33 2023

@author: Bart Steeman. Govers Lab.
"""
#from . import plot
from .imagecollection import ImageCollection
from .logging_utils import configure_bactoscoop_logging

__all__ = ["ImageCollection", "Curation", "configure_bactoscoop_logging"]


def __getattr__(name):
    if name == "Curation":
        from .curation import Curation

        return Curation
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
