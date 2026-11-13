"""BluePixel ComfyUI nodes package.

Import does not load any model weights; heavy initialization happens inside node
execution so that ComfyUI startup stays light.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
