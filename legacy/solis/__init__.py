"""Solis — a sparse Mixture-of-Experts language model, built from scratch."""

from .config import SolisConfig, PRESETS, get_config
from .model import Solis, KVCache, generate_stream
from .tokenizer import SolisTokenizer, load_default as load_tokenizer
from .multimodal import (
    SolisMM,
    VisionConfig,
    AudioConfig,
    DEFAULT_VISION,
    DEFAULT_AUDIO,
)
from .mcp import (
    MCPManager,
    MCPClient,
    ServerConfig,
    Tool,
    ToolResult,
    MCPError,
    render_tools_prompt,
    parse_tool_calls,
)

__all__ = [
    "SolisConfig",
    "PRESETS",
    "get_config",
    "Solis",
    "KVCache",
    "generate_stream",
    "SolisTokenizer",
    "load_tokenizer",
    "SolisMM",
    "VisionConfig",
    "AudioConfig",
    "DEFAULT_VISION",
    "DEFAULT_AUDIO",
    "MCPManager",
    "MCPClient",
    "ServerConfig",
    "Tool",
    "ToolResult",
    "MCPError",
    "render_tools_prompt",
    "parse_tool_calls",
]
__version__ = "1.1.0"
