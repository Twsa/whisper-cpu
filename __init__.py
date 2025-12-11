"""
Whisper to LRC 转换器
使用 faster_whisper 将音频文件转换为带时间戳的 LRC 字幕文件
"""

__version__ = "1.0.0"
__author__ = "Twsa"
__email__ = ""

from .whisper_lrc import WhisperLRCGenerator
from .batch_process import BatchProcessor
from .utils import is_audio_file, format_timestamp_lrc, get_output_filename

__all__ = [
    "WhisperLRCGenerator",
    "BatchProcessor",
    "is_audio_file",
    "format_timestamp_lrc",
    "get_output_filename",
]