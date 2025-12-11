#!/usr/bin/env python3
"""主程序入口 - 使用 faster_whisper 生成 LRC 字幕文件"""
import argparse
import sys
import os
from whisper_lrc import WhisperLRCGenerator
from batch_process import BatchProcessor
from utils import is_audio_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='使用 faster-whisper 将音频转换为 LRC 字幕文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单文件转换
  %(prog)s audio.mp3                    # 基本使用
  %(prog)s audio.mp3 -o out.lrc        # 指定输出文件
  %(prog)s audio.mp3 -m small          # 使用 small 模型
  %(prog)s audio.mp3 -l zh             # 指定语言为中文
  %(prog)s audio.mp3 --vad-filter      # 启用语音活动检测

  # 批处理模式
  %(prog)s --batch /path/to/audio/dir  # 递归批处理目录
  %(prog)s -b /path/to/dir --no-recursive  # 仅处理当前目录
  %(prog)s -b /path/to/dir --force    # 强制覆盖已存在的 LRC 文件
        """
    )

    # 添加互斥组：要么指定单个文件，要么指定批处理
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        'input',
        nargs='?',
        help='输入音频文件路径（单文件模式）'
    )

    group.add_argument(
        '-b', '--batch',
        help='批处理模式：指定要处理的目录路径'
    )

    # 可选参数
    parser.add_argument(
        '-o', '--output',
        help='输出 LRC 文件路径（单文件模式，默认：输入文件同目录下的 .lrc 文件）'
    )

    parser.add_argument(
        '-m', '--model',
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper 模型大小 (默认: base)'
    )

    parser.add_argument(
        '-l', '--language',
        help='语言代码 (例如: zh, en, ja)，默认自动检测'
    )

    parser.add_argument(
        '--beam-size',
        type=int,
        default=5,
        help='Beam search 大小 (默认: 5)'
    )

    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='计算设备 (默认: cpu)'
    )

    parser.add_argument(
        '--compute-type',
        default='int8',
        choices=['int8', 'float16', 'float32'],
        help='计算类型 (默认: int8)'
    )

    parser.add_argument(
        '--vad-filter',
        action='store_true',
        help='启用语音活动检测过滤静音部分'
    )

    # 批处理相关参数
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='批处理模式：不递归处理子目录（默认：递归）'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='批处理模式：强制覆盖已存在的 LRC 文件'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.1.0'
    )

    args = parser.parse_args()

    # 单文件模式
    if args.input:
        # 检查输入文件
        if not os.path.exists(args.input):
            print(f"错误: 音频文件不存在: {args.input}")
            return 1

        # 检查是否为音频文件
        if not is_audio_file(os.path.splitext(args.input)[1]):
            print(f"错误: 不支持的文件格式: {args.input}")
            return 1

        # 创建生成器
        generator = WhisperLRCGenerator(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type
        )

        # 执行转录
        try:
            output_file = generator.transcribe_to_lrc(
                audio_path=args.input,
                output_path=args.output,
                language=args.language,
                beam_size=args.beam_size,
                vad_filter=args.vad_filter
            )
            print(f"\n✓ 成功生成 LRC 文件: {output_file}")
            return 0
        except Exception as e:
            print(f"\n✗ 错误: {str(e)}")
            return 1

    # 批处理模式
    elif args.batch:
        # 检查输入目录
        if not os.path.isdir(args.batch):
            print(f"错误: 目录不存在: {args.batch}")
            return 1

        # 创建批处理器
        processor = BatchProcessor(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type
        )

        # 执行批处理
        try:
            processor.process_directory(
                directory=args.batch,
                recursive=not args.no_recursive,
                language=args.language,
                beam_size=args.beam_size,
                vad_filter=args.vad_filter,
                skip_existing=not args.force
            )
            return 0
        except Exception as e:
            print(f"\n✗ 错误: {str(e)}")
            return 1

if __name__ == "__main__":
    sys.exit(main())