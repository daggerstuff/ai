#!/usr/bin/env python3
"""
Enterprise Upgrade for Existing Components

Upgrades all existing dataset processing components to enterprise baseline:
- Adds enterprise logging
- Adds error handling
- Adds configuration management
- Adds monitoring integration
- Adds performance tracking
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

class EnterpriseComponentUpgrader:
    """Upgrades existing components to enterprise standards."""
    
    def __init__(self):
        self.base_path = Path("/home/vivi/pixelated/ai")
        self.dataset_pipeline_path = self.base_path / "dataset_pipeline"
        
        # Components that need upgrading
        self.components_to_upgrade = [
            "orchestrator_minimal.py",
            "data_standardizer.py", 
            "batch_processor.py",
            "streaming_processor.py",
            "performance_optimizer.py"
        ]
        
        print("🔧 Enterprise Component Upgrader initialized")
    
    def add_enterprise_imports(self, file_path: Path) -> str:
        """Add enterprise imports to a Python file."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if enterprise imports already exist
        if 'enterprise_config' in content:
            return content
        
        # Find the import section
        import_pattern = r'(import\s+\w+.*?\n|from\s+\w+.*?\n)+'
        imports_match = re.search(import_pattern, content, re.MULTILINE)
        
        enterprise_imports = '''
# Enterprise baseline imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "enterprise_config"))

from enterprise_config import get_config
from enterprise_logging import get_logger
from enterprise_error_handling import handle_error, with_retry, with_circuit_breaker
'''
        
        if imports_match:
            # Insert after existing imports
            insert_pos = imports_match.end()
            content = content[:insert_pos] + enterprise_imports + content[insert_pos:]
        else:
            # Insert at the beginning after shebang and docstring
            lines = content.split('\n')
            insert_line = 0
            
            # Skip shebang
            if lines[0].startswith('#!'):
                insert_line = 1
            
            # Skip docstring
            if insert_line < len(lines) and lines[insert_line].strip().startswith('"""'):
                for i in range(insert_line + 1, len(lines)):
                    if lines[i].strip().endswith('"""'):
                        insert_line = i + 1
                        break
            
            lines.insert(insert_line, enterprise_imports)
            content = '\n'.join(lines)
        
        return content
    
    def add_enterprise_initialization(self, file_path: Path, content: str) -> str:
        """Add enterprise initialization to a class."""
        
        # Find class definitions
        class_pattern = r'class\s+(\w+).*?:'
        classes = re.findall(class_pattern, content)
        
        for class_name in classes:
            # Find __init__ method
            init_pattern = rf'(class\s+{class_name}.*?\n.*?def\s+__init__\(self.*?\):.*?\n)(.*?)(?=\n\s*def|\nclass|\Z)'
            init_match = re.search(init_pattern, content, re.DOTALL)
            
            if init_match:
                init_start = init_match.group(1)
                init_body = init_match.group(2)
                
                # Add enterprise initialization
                enterprise_init = '''        
        # Enterprise baseline initialization
        self.config = get_config()
        self.logger = get_logger(self.__class__.__name__.lower())
        self.logger.info(f"Initialized {self.__class__.__name__} with enterprise baseline")
'''
                
                if 'self.config = get_config()' not in init_body:
                    new_init = init_start + enterprise_init + init_body
                    content = content.replace(init_match.group(0), new_init)
        
        return content
    
    def add_error_handling_decorators(self, file_path: Path, content: str) -> str:
        """Add error handling decorators to key methods."""
        
        # Methods that should have error handling
        critical_methods = ['process', 'run', 'execute', 'validate', 'load', 'save']
        
        for method in critical_methods:
            # Find method definitions
            method_pattern = rf'(\s+def\s+{method}[^:]*:)'
            
            def add_decorator(match):
                indent = len(match.group(1)) - len(match.group(1).lstrip())
                decorator = ' ' * (indent - 4) + f'@with_retry(component="{file_path.stem}", strategy="default")\n'
                return decorator + match.group(1)
            
            content = re.sub(method_pattern, add_decorator, content)
        
        return content
    
    def add_performance_logging(self, file_path: Path, content: str) -> str:
        """Add performance logging to key operations."""
        
        # Find processing loops and add performance logging
        performance_patterns = [
            (r'(for\s+\w+\s+in\s+.*?:)', 'processing_loop'),
            (r'(while\s+.*?:)', 'processing_while'),
            (r'(def\s+process.*?:)', 'process_method')
        ]
        
        for pattern, operation_type in performance_patterns:
            def add_perf_logging(match):
                return f'''# Performance logging for {operation_type}
        start_time = time.time()
        {match.group(1)}
            # Log performance every 1000 iterations
            if hasattr(self, '_iteration_count'):
                self._iteration_count += 1
                if self._iteration_count % 1000 == 0:
                    elapsed = time.time() - start_time
                    self.logger.info(f"Processed {{self._iteration_count}} items in {{elapsed:.2f}}s")
            else:
                self._iteration_count = 1'''
            
            content = re.sub(pattern, add_perf_logging, content, count=1)
        
        return content
    
    def upgrade_component(self, component_file: str) -> bool:
        """Upgrade a single component to enterprise standards."""
        file_path = self.dataset_pipeline_path / component_file
        
        if not file_path.exists():
            print(f"   ⚠️ {component_file} not found, skipping")
            return False
        
        print(f"   🔧 Upgrading {component_file}...")
        
        try:
            # Read original content
            with open(file_path, 'r') as f:
                original_content = f.read()
            
            # Create backup
            backup_path = file_path.with_suffix('.py.backup')
            with open(backup_path, 'w') as f:
                f.write(original_content)
            
            # Apply enterprise upgrades
            content = original_content
            content = self.add_enterprise_imports(file_path, content)
            content = self.add_enterprise_initialization(file_path, content)
            content = self.add_error_handling_decorators(file_path, content)
            content = self.add_performance_logging(file_path, content)
            
            # Write upgraded content
            with open(file_path, 'w') as f:
                f.write(content)
            
            print(f"   ✅ {component_file} upgraded (backup: {backup_path.name})")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to upgrade {component_file}: {e}")
            return False
    
    def create_enterprise_wrapper_template(self):
        """Create a template for wrapping existing functions with enterprise features."""
        
        template = '''#!/usr/bin/env python3
"""
Enterprise Wrapper Template

Use this template to wrap existing functions with enterprise features.
"""

import time
import functools
from typing import Any, Callable

# Enterprise imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "enterprise_config"))

from enterprise_config import get_config
from enterprise_logging import get_logger, LogPerformance
from enterprise_error_handling import handle_error, with_retry, with_circuit_breaker

def enterprise_wrapper(component_name: str = "unknown", 
                      enable_retry: bool = True,
                      enable_circuit_breaker: bool = False,
                      enable_performance_logging: bool = True):
    """
    Decorator to add enterprise features to any function.
    
    Args:
        component_name: Name of the component for logging
        enable_retry: Enable automatic retry on failures
        enable_circuit_breaker: Enable circuit breaker protection
        enable_performance_logging: Enable performance logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            config = get_config()
            logger = get_logger(component_name)
            
            # Performance logging
            if enable_performance_logging:
                with LogPerformance(logger, f"{func.__name__}", 
                                  component=component_name):
                    try:
                        if enable_retry:
                            @with_retry(component_name)
                            def retry_func():
                                return func(*args, **kwargs)
                            return retry_func()
                        else:
                            return func(*args, **kwargs)
                    except Exception as e:
                        handle_error(e, component_name, {
                            'function': func.__name__,
                            'args_count': len(args),
                            'kwargs_keys': list(kwargs.keys())
                        })
                        raise
            else:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    handle_error(e, component_name)
                    raise
        
        return wrapper
    return decorator

# Example usage:
@enterprise_wrapper("example_component", enable_retry=True, enable_performance_logging=True)
def example_function(data):
    """Example function with enterprise features."""
    # Your existing code here
    return processed_data

if __name__ == "__main__":
    print("✅ Enterprise wrapper template ready")
    print("Use @enterprise_wrapper() decorator on your functions")
'''
        
        template_path = self.base_path / "enterprise_config" / "enterprise_wrapper_template.py"
        with open(template_path, 'w') as f:
            f.write(template)
        
        print(f"   ✅ Enterprise wrapper template created: {template_path}")
    
    def run_comprehensive_upgrade(self):
        """Run comprehensive enterprise upgrade on all components."""
        print("🚀 STARTING ENTERPRISE COMPONENT UPGRADE")
        print("=" * 60)
        
        upgraded_count = 0
        
        # Upgrade individual components
        print("🔧 Upgrading dataset pipeline components...")
        for component in self.components_to_upgrade:
            if self.upgrade_component(component):
                upgraded_count += 1
        
        # Create enterprise wrapper template
        print("📝 Creating enterprise wrapper template...")
        self.create_enterprise_wrapper_template()
        
        # Create enterprise integration guide
        self.create_integration_guide()
        
        print("=" * 60)
        print(f"🎯 ENTERPRISE UPGRADE COMPLETE")
        print(f"✅ {upgraded_count}/{len(self.components_to_upgrade)} components upgraded")
        print("✅ Enterprise wrapper template created")
        print("✅ Integration guide created")
        
        if upgraded_count == len(self.components_to_upgrade):
            print("🎉 ALL COMPONENTS SUCCESSFULLY UPGRADED TO ENTERPRISE BASELINE!")
        else:
            print("⚠️ Some components need manual attention")
        
        return upgraded_count == len(self.components_to_upgrade)
    
    def create_integration_guide(self):
        """Create integration guide for enterprise features."""
        
        guide = '''# Enterprise Integration Guide

## Quick Start

### 1. Add Enterprise Features to Existing Functions

```python
from enterprise_config.enterprise_wrapper_template import enterprise_wrapper

@enterprise_wrapper("my_component", enable_retry=True)
def my_existing_function(data):
    # Your existing code
    return result
```

### 2. Manual Integration

```python
# Add to imports
from enterprise_config import get_config
from enterprise_logging import get_logger
from enterprise_error_handling import handle_error, with_retry

# Add to class __init__
def __init__(self):
    self.config = get_config()
    self.logger = get_logger(self.__class__.__name__.lower())

# Add error handling to critical methods
@with_retry(component="my_component")
def critical_method(self):
    try:
        # Your code
        pass
    except Exception as e:
        handle_error(e, "my_component")
        raise
```

### 3. Configuration Usage

```python
config = get_config()
batch_size = config.batch_size
quality_threshold = config.quality_threshold
```

### 4. Logging Usage

```python
logger = get_logger("my_component")
logger.info("Processing started")
logger.error("Error occurred", extra={'context': 'additional_info'})
```

### 5. Performance Monitoring

```python
from enterprise_logging import LogPerformance

with LogPerformance(logger, "expensive_operation"):
    # Your expensive operation
    result = process_large_dataset()
```

## Enterprise Standards Checklist

- [ ] Configuration management integrated
- [ ] Structured logging implemented
- [ ] Error handling with retry logic
- [ ] Performance monitoring added
- [ ] Health checks implemented
- [ ] Security considerations addressed
- [ ] Documentation updated
- [ ] Tests include enterprise features

## Best Practices

1. **Always use configuration instead of hardcoded values**
2. **Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)**
3. **Handle errors gracefully with context**
4. **Monitor performance of critical operations**
5. **Implement health checks for all components**
6. **Use circuit breakers for external dependencies**
7. **Follow security guidelines for sensitive data**

## Troubleshooting

### Import Errors
If you get import errors, ensure the enterprise_config directory is in your Python path:

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "enterprise_config"))
```

### Configuration Issues
Check configuration with:

```bash
python enterprise_config/status_dashboard.py
```

### Logging Issues
Check log files in the `logs/` directory.
'''
        
        guide_path = self.base_path / "enterprise_config" / "INTEGRATION_GUIDE.md"
        with open(guide_path, 'w') as f:
            f.write(guide)
        
        print(f"   ✅ Integration guide created: {guide_path}")

if __name__ == "__main__":
    upgrader = EnterpriseComponentUpgrader()
    
    # Run comprehensive upgrade
    success = upgrader.run_comprehensive_upgrade()
    
    print("\n🏢 Enterprise component upgrade complete!")
    print("📖 See enterprise_config/INTEGRATION_GUIDE.md for usage instructions")
    
    if success:
        print("✅ All components ready for enterprise-grade operation!")
    else:
        print("⚠️ Some components may need manual review")
