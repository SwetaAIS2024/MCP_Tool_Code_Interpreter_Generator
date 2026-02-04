"""Centralized logging configuration for the pipeline.

Supports multiple log levels and outputs:
- Console: Clean, minimal output by default
- File: Detailed logs for debugging
- Verbosity flags: --quiet, --verbose, --debug
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class PipelineLogger:
    """Configure logging for the entire pipeline."""
    
    _instance: Optional['PipelineLogger'] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not PipelineLogger._initialized:
            self.log_dir = Path("logs")
            self.log_dir.mkdir(exist_ok=True)
            self.verbosity = "normal"  # quiet, normal, verbose, debug
            PipelineLogger._initialized = True
    
    def setup(self, verbosity: str = "normal", log_file: Optional[str] = None):
        """Configure logging with specified verbosity.
        
        Args:
            verbosity: One of 'quiet', 'normal', 'verbose', 'debug'
            log_file: Optional custom log file path
        """
        self.verbosity = verbosity
        
        # Remove existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set root logger level
        root_logger.setLevel(logging.DEBUG)
        
        # Console handler with verbosity-based level
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._get_console_formatter())
        
        if verbosity == "quiet":
            console_handler.setLevel(logging.WARNING)
        elif verbosity == "normal":
            console_handler.setLevel(logging.INFO)
        elif verbosity == "verbose":
            console_handler.setLevel(logging.INFO)
        else:  # debug
            console_handler.setLevel(logging.DEBUG)
        
        root_logger.addHandler(console_handler)
        
        # File handler - always logs everything
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"pipeline_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self._get_file_formatter())
        root_logger.addHandler(file_handler)
        
        # Log the configuration
        logger = logging.getLogger(__name__)
        logger.debug(f"Logging configured: verbosity={verbosity}, file={log_file}")
    
    def _get_console_formatter(self):
        """Get formatter for console output (clean, minimal)."""
        if self.verbosity == "debug":
            return logging.Formatter(
                '%(levelname)s [%(name)s] %(message)s'
            )
        else:
            # Clean format for normal/verbose - no logger names
            return logging.Formatter('%(message)s')
    
    def _get_file_formatter(self):
        """Get formatter for file output (detailed)."""
        return logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Configured logger
    """
    return logging.getLogger(name)


# Convenience functions for common logging patterns
def log_section(logger: logging.Logger, title: str, char: str = "="):
    """Log a section header."""
    logger.info(f"\n{char * 80}")
    logger.info(title)
    logger.info(f"{char * 80}\n")


def log_subsection(logger: logging.Logger, title: str):
    """Log a subsection header."""
    logger.info(f"\n{title}")
    logger.info("-" * 80)


def log_step(logger: logging.Logger, step: str, emoji: str = "📍"):
    """Log a pipeline step."""
    logger.info(f"{emoji} {step}")


def log_success(logger: logging.Logger, message: str):
    """Log a success message."""
    logger.info(f"✅ {message}")


def log_error(logger: logging.Logger, message: str):
    """Log an error message."""
    logger.error(f"❌ {message}")


def log_warning(logger: logging.Logger, message: str):
    """Log a warning message."""
    logger.warning(f"⚠️  {message}")
