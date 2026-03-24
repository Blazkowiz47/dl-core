import logging
import torch
import os
import sys
import warnings


def _suppress_third_party_logs():
    """Suppress noisy third-party library logs."""
    os.environ["NCCL_DEBUG"] = "WARN"
    logging.getLogger("azure.storage").setLevel(logging.ERROR)
    logging.getLogger("azure").setLevel(logging.ERROR)
    logging.getLogger("azure.core").setLevel(logging.ERROR)
    logging.getLogger("azure.identity").setLevel(logging.ERROR)
    logging.getLogger("azure.identity._internal.decorators").setLevel(logging.ERROR)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
        logging.ERROR
    )

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("torchvision").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
    logging.getLogger("azureml").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    # Filter out common noisy warnings
    warnings.filterwarnings(
        "ignore", message="Connection pool is full, discarding connection"
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


# Suppress third-party logs immediately on module import
_suppress_third_party_logs()


class RankFilter(logging.Filter):
    """
    Logging filter to only show logs from rank 0 in distributed training.

    This filter checks the RANK environment variable dynamically on each log record,
    allowing it to work even when RANK is set after logger initialization.
    In single-GPU or CPU mode (RANK not set), defaults to rank 0 and allows all logs.
    """

    def __init__(self, filter_none: bool = False):
        super().__init__()
        self.filter_none = filter_none

    def filter(self, record):
        """
        Determine if the log record should be processed.

        Args:
            record: LogRecord instance

        Returns:
            True if rank 0 or RANK not set, False otherwise
        """
        rank = int(os.environ.get("RANK", "0"))
        if self.filter_none:
            return True
        return rank == 0


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
        "BOLD": "\033[1m",  # Bold
        "DIM": "\033[2m",  # Dim
    }

    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[levelname]}{self.COLORS['BOLD']}{levelname:8}{self.COLORS['RESET']}"
            record.levelname = colored_levelname

        # Format the message
        formatted = super().format(record)

        # Reset levelname for other formatters
        record.levelname = levelname

        return formatted


class RankFormatter(logging.Formatter):
    """Formatter that dynamically adds rank information to each log record."""

    def format(self, record):
        # Dynamically get rank for each log record
        is_distributed = (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        if is_distributed:
            rank = int(os.environ.get("RANK", "0"))
            record.rank_info = f"[RANK {rank}] "
        else:
            record.rank_info = ""

        return super().format(record)


def setup_logging(level, log_file=None):
    """
    Setup centralized logging configuration for the project.

    Uses RankFilter to ensure only rank 0 logs in distributed training.
    This works regardless of when the RANK environment variable is set.

    Args:
        level: Logging level (default: logging.INFO)
        log_file: Optional file path for logging output

    Returns:
        Root logger instance
    """
    # Check if terminal supports colors
    supports_color = (
        hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
        and os.environ.get("TERM") != "dumb"
    )

    # Create formatter with dynamic rank info
    if supports_color:
        # Create a custom formatter class that combines both ColoredFormatter and RankFormatter
        class ColoredRankFormatter(RankFormatter, ColoredFormatter):
            pass

        console_formatter = ColoredRankFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(rank_info)s%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        console_formatter = RankFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(rank_info)s%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    # Add rank filter to only show logs from rank 0 in multi-GPU training
    console_handler.addFilter(RankFilter(level == "DEBUG"))

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add console handler
    root_logger.addHandler(console_handler)

    # Add file handler if specified (always use plain formatter for files)
    if log_file:
        file_formatter = RankFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(rank_info)s%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        # Add rank filter to only write logs from rank 0 in multi-GPU training
        file_handler.addFilter(RankFilter())
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party libraries
    _suppress_third_party_logs()

    return root_logger


def log_system_info():
    """Log useful system information at startup."""
    logger = logging.getLogger(__name__)

    logger.info("🖥️  System Information:")
    logger.info(f"   • Python: {sys.version.split()[0]}")
    logger.info(f"   • Platform: {sys.platform}")
    logger.info(f"   • Working Directory: {os.getcwd()}")

    # GPU info if available
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"   • GPU: {gpu_name} ({gpu_count} device(s))")
        else:
            logger.info("   • GPU: Not available (CPU only)")
    except ImportError:
        logger.info("   • GPU: PyTorch not available")


def log_experiment_banner(config: dict):
    """Log a beautiful experiment banner with key training information."""
    logger = logging.getLogger(__name__)

    # Create banner
    banner_width = 70
    logger.info("=" * banner_width)
    logger.info("🚀 EXPERIMENT".center(banner_width))
    logger.info("=" * banner_width)

    # Key experiment info
    logger.info("📝 Task:           Presentation Attack Detection")
    logger.info(
        f"🤖 Model:          {config.get('model', {})[0].get('name', 'Unknown')}"
    )
    logger.info(
        f"📊 Dataset:        {config.get('dataset', {}).get('name', 'Unknown')}"
    )
    logger.info(f"⚙️  Device:         {config.get('device', 'Unknown')}")
    logger.info(
        f"🔄 Epochs:         {config.get('training', {}).get('epochs', 'Unknown')}"
    )
    logger.info(
        f"📦 Batch Size:     {config.get('training', {}).get('batch_size', 'Unknown')}"
    )
    logger.info(
        f"🚀 Accelerator:    {config.get('runtime', {}).get('accelerator', {}).get('type', 'CPU')}"
    )

    # Additional training details
    training_config = config.get("training", {})
    if training_config.get("validate_after_epochs"):
        logger.info(
            f"✅ Validation:     Every {training_config['validate_after_epochs']} epoch(s)"
        )

    if config.get("dry_run"):
        logger.info("🧪 Mode:           DRY RUN (Test Mode)")

    logger.info("=" * banner_width)
