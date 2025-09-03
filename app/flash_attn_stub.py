import sys
import types

# provide a no-op flash_attn module so CPU environments avoid optional GPU dependency
sys.modules.setdefault("flash_attn", types.ModuleType("flash_attn"))
