"""批处理模块 - 递归处理目录下的所有音频文件"""
import os
import sys
import time
from pathlib import Path
from whisper_lrc import WhisperLRCGenerator
from utils import get_output_filename, is_audio_file

class BatchProcessor:
    """批处理器，用于递归转换目录下的音频文件"""

    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """初始化批处理器

        Args:
            model_size (str): Whisper 模型大小
            device (str): 计算设备
            compute_type (str): 计算类型
        """
        self.generator = WhisperLRCGenerator(
            model_size=model_size,
            device=device,
            compute_type=compute_type
        )
        self.processed_count = 0
        self.skipped_count = 0
        self.error_count = 0

    def find_audio_files(self, directory, recursive=True):
        """查找目录下的所有音频文件

        Args:
            directory (str): 要搜索的目录
            recursive (bool): 是否递归搜索子目录

        Yields:
            tuple: (音频文件路径, 对应的 LRC 文件路径)
        """
        directory = Path(directory).resolve()

        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in directory.glob(pattern):
            if file_path.is_file() and is_audio_file(file_path.suffix.lower()):
                # 生成对应的 LRC 文件路径（与音频文件同级目录）
                lrc_path = file_path.with_suffix('.lrc')
                yield file_path, lrc_path

    def process_file(self, audio_path, lrc_path, language=None, beam_size=5,
                    vad_filter=False, skip_existing=True):
        """处理单个音频文件

        Args:
            audio_path (Path): 音频文件路径
            lrc_path (Path): LRC 输出路径
            language (str, optional): 语言代码
            beam_size (int): Beam search 大小
            vad_filter (bool): 是否使用 VAD
            skip_existing (bool): 是否跳过已存在的文件

        Returns:
            bool: 是否成功处理
        """
        # 检查是否需要跳过已存在的文件
        if skip_existing and lrc_path.exists():
            print(f"跳过已存在的文件: {lrc_path}")
            self.skipped_count += 1
            return False

        try:
            print(f"\n正在处理: {audio_path}")
            print(f"输出到: {lrc_path}")

            # 转换为字符串路径
            audio_str = str(audio_path)
            lrc_str = str(lrc_path)

            # 执行转录
            self.generator.transcribe_to_lrc(
                audio_path=audio_str,
                output_path=lrc_str,
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter
            )

            self.processed_count += 1
            print(f"✓ 成功转换: {audio_path}")
            return True

        except Exception as e:
            self.error_count += 1
            print(f"✗ 转换失败: {audio_path}")
            print(f"  错误: {str(e)}")
            return False

    def process_directory(self, directory, recursive=True, language=None,
                         beam_size=5, vad_filter=False, skip_existing=True):
        """批处理目录下的音频文件

        Args:
            directory (str): 要处理的目录
            recursive (bool): 是否递归处理子目录
            language (str, optional): 语言代码
            beam_size (int): Beam search 大小
            vad_filter (bool): 是否使用 VAD
            skip_existing (bool): 是否跳过已存在的文件
        """
        start_time = time.time()

        # 查找所有音频文件
        audio_files = list(self.find_audio_files(directory, recursive))

        if not audio_files:
            print(f"在目录 {directory} 中未找到音频文件")
            return

        print(f"找到 {len(audio_files)} 个音频文件")
        print(f"模式: {'递归' if recursive else '仅当前目录'}")
        if language:
            print(f"指定语言: {language}")
        print("-" * 60)

        # 处理每个文件
        for i, (audio_path, lrc_path) in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}]", end=" ")
            self.process_file(
                audio_path, lrc_path, language, beam_size,
                vad_filter, skip_existing
            )

        # 打印统计信息
        elapsed_time = time.time() - start_time
        self.print_statistics(len(audio_files), elapsed_time)

    def print_statistics(self, total_files, elapsed_time):
        """打印处理统计信息"""
        print("\n" + "=" * 60)
        print("批处理完成！")
        print(f"总文件数: {total_files}")
        print(f"成功转换: {self.processed_count}")
        print(f"跳过文件: {self.skipped_count}")
        print(f"错误文件: {self.error_count}")
        print(f"耗时: {elapsed_time:.2f} 秒")
        if self.processed_count > 0:
            avg_time = elapsed_time / self.processed_count
            print(f"平均每文件: {avg_time:.2f} 秒")


def main():
    """批处理模式的命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='批处理模式：递归处理目录下的所有音频文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s /path/to/music              # 递归处理目录
  %(prog)s /path/to/music --no-recursive  # 仅处理当前目录
  %(prog)s /path/to/music -l zh         # 指定中文
  %(prog)s /path/to/music -m tiny       # 使用 tiny 模型
  %(prog)s /path/to/music --force       # 强制覆盖已存在的文件
        """
    )

    parser.add_argument('directory', help='要处理的音频文件目录路径')
    parser.add_argument('-m', '--model', default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper 模型大小 (默认: base)')
    parser.add_argument('-l', '--language',
                       help='指定语言代码 (如: zh, en, ja, ko 等)')
    parser.add_argument('--beam-size', type=int, default=5,
                       help='Beam search 大小 (默认: 5)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='计算设备 (默认: cpu)')
    parser.add_argument('--compute-type', default='int8',
                       choices=['int8', 'float16', 'float32'],
                       help='计算类型 (默认: int8)')
    parser.add_argument('--vad-filter', action='store_true',
                       help='启用语音活动检测（过滤静音部分）')
    parser.add_argument('--no-recursive', action='store_true',
                       help='不递归处理子目录')
    parser.add_argument('--force', action='store_true',
                       help='强制覆盖已存在的 LRC 文件')

    args = parser.parse_args()

    # 检查目录是否存在
    if not os.path.exists(args.directory):
        print(f"错误: 目录不存在: {args.directory}")
        sys.exit(1)

    if not os.path.isdir(args.directory):
        print(f"错误: 不是目录: {args.directory}")
        sys.exit(1)

    # 创建批处理器
    processor = BatchProcessor(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type
    )

    # 开始批处理
    processor.process_directory(
        directory=args.directory,
        recursive=not args.no_recursive,
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        skip_existing=not args.force
    )