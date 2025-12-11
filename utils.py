"""工具函数模块"""
import os

def format_timestamp_lrc(seconds):
    """将秒数转换为 LRC 时间戳格式 [mm:ss.xx]

    Args:
        seconds (float): 秒数

    Returns:
        str: 格式化后的时间戳，例如 [01:31.47]
    """
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"[{minutes:02d}:{remaining_seconds:05.2f}]"

def get_output_filename(input_filename):
    """根据输入音频文件名生成输出 LRC 文件名

    Args:
        input_filename (str): 输入文件名

    Returns:
        str: 输出的 LRC 文件名
    """
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    return f"{base_name}.lrc"

def ensure_output_directory(output_path):
    """确保输出目录存在

    Args:
        output_path (str): 输出文件路径
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

def is_audio_file(file_extension):
    """判断文件是否为音频文件

    Args:
        file_extension (str): 文件扩展名（包含点号）

    Returns:
        bool: 是否为音频文件
    """
    audio_extensions = {
        '.mp3', '.mp4', '.wav', '.m4a', '.flac', '.aac', '.ogg',
        '.wma', '.opus', '.mp2', '.m4b', '.m4p', '.webm', '.mkv',
        '.avi', '.mov', '.wmv', '.flv', '.3gp'
    }
    return file_extension.lower() in audio_extensions