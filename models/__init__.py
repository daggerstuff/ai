"""Pixel model stubs."""

# PixelBaseModel moved to models/
try:
    from models.base.pixel_base_model import PixelBaseModel
except ImportError:
    PixelBaseModel = None

__all__ = ["PixelBaseModel"]
