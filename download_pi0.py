#!/usr/bin/env python3
"""
pi0_base 模型下载工具
基于 OpenPI 原始下载逻辑，直接使用 gs:// 链接通过 fsspec 下载
具备缓存机制、文件锁、权限管理等功能
"""

import argparse
import concurrent.futures
import datetime
import logging
import os
import pathlib
import re
import shutil
import stat
import sys
import time
import urllib.parse
from typing import Optional

import filelock
import tqdm

try:
    import fsspec
    import fsspec.core
    FSSPEC_AVAILABLE = True
except ImportError:
    FSSPEC_AVAILABLE = False
    print("⚠️ 警告: fsspec 未安装，将尝试使用备用方法")

# 环境变量控制缓存目录路径，默认使用 ~/.cache/openpi
_OPENPI_DATA_HOME = "OPENPI_DATA_HOME"

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 缓存过期时间映射
_INVALIDATE_CACHE_DIRS = {
    re.compile("openpi-assets/checkpoints/pi0_aloha_pen_uncap"): datetime.datetime(2025, 2, 17),
    re.compile("openpi-assets/checkpoints/pi0_libero"): datetime.datetime(2025, 2, 6),
    re.compile("openpi-assets/checkpoints/"): datetime.datetime(2025, 2, 3),
}


def get_cache_dir() -> pathlib.Path:
    """获取缓存目录"""
    default_dir = pathlib.Path.home() / ".cache" / "openpi"

    cache_dir = pathlib.Path(
        os.getenv(_OPENPI_DATA_HOME, str(default_dir))).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_folder_permission(cache_dir)
    return cache_dir


def _set_permission(path: pathlib.Path, target_permission: int):
    """设置文件权限"""
    if not path.exists():
        return

    if path.stat().st_mode & target_permission == target_permission:
        logger.debug(f"跳过 {path}，权限已正确")
        return

    try:
        path.chmod(target_permission)
        logger.debug(f"设置 {path} 权限为 {target_permission}")
    except PermissionError as e:
        logger.warning(f"无法设置权限 {path}: {e}")


def _set_folder_permission(folder_path: pathlib.Path) -> None:
    """设置文件夹权限为可读写搜索"""
    _set_permission(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


def _ensure_permissions(path: pathlib.Path) -> None:
    """确保缓存目录具有正确的权限"""
    def _setup_folder_permission_between_cache_dir_and_path(path: pathlib.Path) -> None:
        cache_dir = get_cache_dir()
        try:
            relative_path = path.relative_to(cache_dir)
            moving_path = cache_dir
            for part in relative_path.parts:
                _set_folder_permission(moving_path / part)
                moving_path = moving_path / part
        except ValueError:
            # 如果路径不在缓存目录中，跳过
            pass

    def _set_file_permission(file_path: pathlib.Path) -> None:
        """设置文件为可读写，如果是脚本则保持可执行"""
        file_rw = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
        if file_path.stat().st_mode & 0o100:
            _set_permission(file_path, file_rw | stat.S_IXUSR |
                            stat.S_IXGRP | stat.S_IXOTH)
        else:
            _set_permission(file_path, file_rw)

    if not path.exists():
        return

    _setup_folder_permission_between_cache_dir_and_path(path)

    if path.is_file():
        _set_file_permission(path)
    else:
        for root, dirs, files in os.walk(str(path)):
            root_path = pathlib.Path(root)
            for file in files:
                file_path = root_path / file
                _set_file_permission(file_path)
            for dir in dirs:
                dir_path = root_path / dir
                _set_folder_permission(dir_path)


def _should_invalidate_cache(cache_dir: pathlib.Path, local_path: pathlib.Path) -> bool:
    """检查缓存是否应该失效"""
    if not local_path.exists():
        return False

    try:
        relative_path = str(local_path.relative_to(cache_dir))
        for pattern, expire_time in _INVALIDATE_CACHE_DIRS.items():
            if pattern.match(relative_path):
                # 如果文件时间早于过期时间，则需要重新下载
                return local_path.stat().st_mtime <= time.mktime(expire_time.timetuple())
    except ValueError:
        # 如果路径不在缓存目录中，不需要失效
        pass

    return False


def _download_fsspec(url: str, local_path: pathlib.Path, **kwargs) -> None:
    """使用 fsspec 下载文件，仿照原始实现"""
    if not FSSPEC_AVAILABLE:
        raise ImportError(
            "fsspec 未安装，无法下载 gs:// 链接。请运行: pip install fsspec[gcs] 或 pip install gcsfs")

    try:
        fs, _ = fsspec.core.url_to_fs(url, **kwargs)
        info = fs.info(url)

        # 文件夹用 0 字节对象表示，末尾有斜杠
        if is_dir := (info["type"] == "directory" or (info["size"] == 0 and info["name"].endswith("/"))):
            total_size = fs.du(url)
        else:
            total_size = info["size"]

        with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(fs.get, url, local_path, recursive=is_dir)
            while not future.done():
                if local_path.exists():
                    current_size = sum(f.stat().st_size for f in [
                                       *local_path.rglob("*"), local_path] if f.is_file())
                    pbar.update(current_size - pbar.n)
                time.sleep(1)
            pbar.update(total_size - pbar.n)

    except ImportError as e:
        if "gcsfs" in str(e).lower():
            raise ImportError(
                "需要安装 gcsfs 来访问 Google Storage。请运行:\n"
                "pip install gcsfs\n"
                "或者:\n"
                "pip install fsspec[gcs]"
            ) from e
        else:
            raise


def maybe_download(url: str, *, force_download: bool = False, use_cache: bool = True, **kwargs) -> pathlib.Path:
    """
    下载文件或目录到本地缓存，并返回本地路径（仿照原始实现）

    如果本地文件已存在，将直接返回。支持并发安全访问。

    Args:
        url: 要下载的文件URL (支持 gs://, http://, https:// 等)
        force_download: 如果为True，即使文件已存在也会重新下载
        use_cache: 是否使用缓存机制
        **kwargs: 传递给 fsspec 的额外参数

    Returns:
        下载文件的本地路径，保证存在且为绝对路径
    """
    # 解析URL
    parsed = urllib.parse.urlparse(url)

    # 如果是本地路径，直接返回
    if parsed.scheme == "":
        path = pathlib.Path(url)
        if not path.exists():
            raise FileNotFoundError(f"文件未找到: {url}")
        return path.resolve()

    if not use_cache:
        # 如果不使用缓存，直接下载到临时文件
        import tempfile
        temp_dir = pathlib.Path(tempfile.mkdtemp())
        filename = pathlib.Path(parsed.path).name or "downloaded_file"
        local_path = temp_dir / filename
        _download_fsspec(url, local_path, **kwargs)
        return local_path

    # 使用缓存机制
    cache_dir = get_cache_dir()
    local_path = cache_dir / parsed.netloc / parsed.path.strip("/")
    local_path = local_path.resolve()

    # 检查缓存是否应该失效
    invalidate_cache = False
    if local_path.exists():
        if force_download or _should_invalidate_cache(cache_dir, local_path):
            invalidate_cache = True
        else:
            print(f"📂 使用缓存文件: {local_path}")
            return local_path

    try:
        # 使用文件锁确保并发安全
        lock_path = local_path.with_suffix(".lock")
        with filelock.FileLock(str(lock_path)):
            # 确保锁文件权限
            _ensure_permissions(lock_path)

            # 如果缓存过期，先删除旧文件
            if invalidate_cache:
                logger.info(f"删除过期缓存: {local_path}")
                if local_path.is_dir():
                    shutil.rmtree(local_path)
                else:
                    local_path.unlink()

            # 下载到临时文件
            logger.info(f"下载 {url} 到 {local_path}")
            scratch_path = local_path.with_suffix(".partial")
            _download_fsspec(url, scratch_path, **kwargs)

            # 移动到最终位置
            shutil.move(scratch_path, local_path)
            _ensure_permissions(local_path)

    except PermissionError as e:
        msg = (
            f"下载时遇到权限错误: {url}\n"
            f"请尝试删除缓存数据后重试: rm -rf {local_path}*"
        )
        raise PermissionError(msg) from e

    return local_path


def download_pi0_model(
    model_name: str = "pi0_base",
    output_dir: Optional[str] = None,
    force: bool = False,
    use_cache: bool = True,
    **kwargs
) -> pathlib.Path:
    """
    下载 pi0 模型

    Args:
        model_name: 模型名称
        output_dir: 输出目录（如果为None，则使用缓存）
        force: 是否强制重新下载
        use_cache: 是否使用缓存机制
        **kwargs: 传递给 fsspec 的额外参数

    Returns:
        下载文件的本地路径
    """
    # 构建 GS 链接
    gs_url = f"gs://openpi-assets/checkpoints/{model_name}"
    if model_name == "droid_100":
        gs_url = f"gs://gresearch/robotics/droid_100"

    if model_name == "google":
        gs_url = f"gs://big_vision/paligemma_tokenizer.model"

    print(f"🔗 模型链接: {gs_url}")

    if output_dir is None:
        # 使用缓存机制
        return maybe_download(gs_url, force_download=force, use_cache=use_cache, **kwargs)
    else:
        # 下载到指定目录
        output_path = pathlib.Path(output_dir) / model_name

        # 检查文件是否已存在
        if output_path.exists() and not force:
            print(f"📂 文件已存在: {output_path}")
            print("💡 使用 --force 参数强制重新下载")
            return output_path

        # 创建输出目录
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 下载文件
        try:
            print(f"📁 下载到: {output_path}")
            _download_fsspec(gs_url, output_path, **kwargs)
            return output_path
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            # 清理部分下载的文件
            if output_path.exists():
                if output_path.is_dir():
                    shutil.rmtree(output_path)
                else:
                    output_path.unlink()
            raise


def extract_model(archive_path: pathlib.Path, extract_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """解压模型文件"""
    if extract_dir is None:
        extract_dir = archive_path.parent / archive_path.stem

    print(f"📦 解压到: {extract_dir}")

    # 创建解压目录
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        if archive_path.name.endswith('.tar.gz') or archive_path.name.endswith('.tgz'):
            import tarfile
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
        elif archive_path.suffix == '.zip':
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            print(f"⚠️ 不支持的文件格式: {archive_path.suffix}")
            return archive_path
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return archive_path

    print(f"✅ 解压完成: {extract_dir}")
    return extract_dir


def install_dependencies():
    """安装必要的依赖"""
    print("📦 检查依赖...")

    try:
        import fsspec
        print("✅ fsspec 已安装")
    except ImportError:
        print("❌ fsspec 未安装")
        print("请运行: pip install fsspec")
        return False

    try:
        import gcsfs
        print("✅ gcsfs 已安装")
        return True
    except ImportError:
        print("❌ gcsfs 未安装")
        print("请运行以下命令之一:")
        print("  pip install gcsfs")
        print("  pip install fsspec[gcs]")
        return False


def _resolve_model_output(output: str | None, no_cache: bool) -> str | None:
    """Choose cached or direct output semantics for predefined models."""
    if output is not None:
        return output
    if no_cache:
        return "./checkpoints"
    return None


def main():
    parser = argparse.ArgumentParser(description="pi0 模型下载工具")
    parser.add_argument(
        "--model",
        default="pi0_base",
        choices=[
            "pi0_base", "pi0_fast_base", "pi0_fast_droid", "pi0_droid",
            "pi0_aloha_towel", "pi0_aloha_tupperware", "pi0_aloha_pen_uncap", "pi0_fast_libero", "pi0_aloha_sim", "pi0_libero", "droid_100", "google", "pi05_libero"
        ],
        help="要下载的模型名称"
    )
    parser.add_argument(
        "--output", "-o", help="输出目录（如果不指定，使用缓存）", default=None)
    parser.add_argument("--force", "-f", action="store_true", help="强制重新下载")
    parser.add_argument("--extract", "-x",
                        action="store_true", help="自动解压下载的文件")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="不使用缓存机制；未指定 --output 时下载到 ./checkpoints",
    )
    parser.add_argument("--gs-url", help="自定义 GS 链接")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--check-deps", action="store_true", help="检查依赖安装")
    parser.add_argument("--token", help="GCS 认证 token（可选）")

    args = parser.parse_args()

    # 检查依赖
    if args.check_deps:
        success = install_dependencies()
        sys.exit(0 if success else 1)

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 准备 fsspec 参数
    fsspec_kwargs = {}
    if args.token:
        fsspec_kwargs['token'] = args.token

    try:
        if args.gs_url:
            # 使用自定义 GS 链接
            output_path = maybe_download(
                args.gs_url, force_download=args.force, use_cache=not args.no_cache, **fsspec_kwargs)
        else:
            # 下载预定义模型
            output_path = download_pi0_model(
                args.model,
                _resolve_model_output(args.output, args.no_cache),
                args.force,
                use_cache=not args.no_cache,
                **fsspec_kwargs
            )

        print(f"📁 文件路径: {output_path}")

        # 自动解压
        if args.extract:
            extract_model(output_path)

        print(f"🎉 任务完成!")

    except ImportError as e:
        print(f"❌ 依赖错误: {e}")
        print("\n💡 解决方案:")
        print("pip install gcsfs")
        print("或者:")
        print("pip install fsspec[gcs]")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ 下载被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    # export HF_ENDPOINT=https://hf-mirror.com
