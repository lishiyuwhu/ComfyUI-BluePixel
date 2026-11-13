"""BluePixel ComfyUI nodes package.

ComfyUI loads this file from custom_nodes/<repo>/__init__.py and reads
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS. Import does NOT load any
model weights; all heavy initialization happens inside node execution.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
