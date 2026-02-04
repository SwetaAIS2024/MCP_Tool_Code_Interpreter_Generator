# Logging System

## Overview

The pipeline now uses a centralized logging system with configurable verbosity levels. Logs are written to both console and files.

## Usage

### Command Line Flags

```bash
# Normal mode (default) - Shows key milestones only
python test_pipeline_with_feedback.py

# Quiet mode - Only warnings and errors
python test_pipeline_with_feedback.py --quiet
python test_pipeline_with_feedback.py -q

# Verbose mode - Shows all information
python test_pipeline_with_feedback.py --verbose
python test_pipeline_with_feedback.py -v

# Debug mode - Shows everything including internals
python test_pipeline_with_feedback.py --debug
python test_pipeline_with_feedback.py -d

# Auto-approve mode (for testing)
python test_pipeline_with_feedback.py --auto
```

### Log Files

All runs generate a detailed log file in `logs/pipeline_{timestamp}.log` containing:
- Full execution trace
- All print statements and debug info
- Timestamps for each operation
- Module and function names

### In Code

```python
from src.logger_config import get_logger, log_section, log_success

logger = get_logger(__name__)

# Use logger instead of print
logger.info("Processing complete")
logger.debug("Debug info")
logger.warning("Warning message")
logger.error("Error occurred")

# Convenience functions
log_section(logger, "SECTION TITLE")
log_success(logger, "Operation completed")
```

## Verbosity Levels

| Level   | Console Output | File Output | Use Case |
|---------|---------------|-------------|----------|
| `quiet` | ⚠️ ❌ only | Everything | Production, when you only care about problems |
| `normal` | ✅ 📍 Key steps | Everything | Default, clean overview of progress |
| `verbose` | All info | Everything | When you want to see what's happening |
| `debug` | Everything | Everything | Troubleshooting, development |

## Log Output Examples

### Normal Mode (Clean)
```
📊 Graph visualization saved to: docs\pipeline_graph.png
📍 Intent extracted
📍 Spec generated  
📍 Code generated
✅ Tool promoted to registry
```

### Debug Mode (Detailed)
```
DEBUG [intent_extraction] Extracting intent from query...
DEBUG [code_generator] Using template: config/prompts/code_generation.txt
INFO [validator] Schema OK: True, Tests OK: True
DEBUG [executor] Module contents: ['FastMCP', 'analyze_...']
```

### Log File (Always Detailed)
```
2026-02-04 16:30:15 - INFO - pipeline - build_graph:73 - Graph visualization saved
2026-02-04 16:30:16 - DEBUG - intent_extraction - intent_node:105 - Query: Run ANOVA...
2026-02-04 16:30:18 - INFO - spec_generator - spec_generator_node:87 - Spec generated
```

## Benefits

1. **Clean Terminal**: Default mode shows only what matters
2. **Full Audit Trail**: Everything saved to log files for debugging
3. **Flexible**: Choose verbosity based on your needs
4. **Professional**: Standard logging practices, easy to extend
5. **Debuggable**: Can enable debug mode when issues occur

## Migration Notes

Modules updated to use logging (gradually migrating all `print()` statements):
- ✅ `test_pipeline_with_feedback.py` - CLI arguments and logger setup
- ⏳ Other modules - Will be migrated incrementally

For now, existing `print()` statements will still work but won't respect verbosity flags.
