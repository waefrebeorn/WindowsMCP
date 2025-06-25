# Vendored DAC nn module
# Exports can be added here if layers.py or quantize.py define classes
# that need to be accessible directly via "from ..nn import X" by other
# parts of the dac_codec, though direct imports like "from ..nn.layers import Y"
# are more common from within dac_codec/model files.

# from .layers import WNConv1d, WNConvTranspose1d, Snake1d # Example
# from .quantize import ResidualVectorQuantize, VectorQuantize # Example

# For now, keeping it simple. The model files will use relative imports.
pass
