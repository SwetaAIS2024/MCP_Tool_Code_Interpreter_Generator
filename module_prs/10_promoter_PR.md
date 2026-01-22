# Module PR 10: Promoter

**Module**: `src/promoter.py`  
**Priority**: P0 (Registry management)  
**Estimated Effort**: 2 days  
**Dependencies**: `01_data_models`

---

## 1. Module Purpose

The Promoter moves approved tools from staging to active registry:
- **Copy** - Move tool file to active directory
- **Register** - Update registry metadata
- **Version** - Handle version conflicts
- **Idempotency** - Safe to run multiple times

**Key Principle**: Only promote after user approval. Never auto-promote.

---

## 2. Core Components

```python
class ToolPromoter:
    """Promote tools from staging to active registry."""
    
    def __init__(
        self,
        staging_dir: Path,
        active_dir: Path,
        registry_file: Path
    ):
        self.staging_dir = staging_dir
        self.active_dir = active_dir
        self.registry_file = registry_file
    
    def promote(self, tool_name: str) -> PromotionResult:
        """
        Promote tool to active registry.
        
        Args:
            tool_name: Name of tool to promote
        
        Returns:
            PromotionResult with success status
        """
        pass
    
    def _copy_tool_file(self, tool_name: str) -> Path:
        """Copy tool from staging to active."""
        pass
    
    def _update_registry(self, tool_name: str, tool_path: Path) -> None:
        """Update registry metadata."""
        pass
    
    def _handle_version_conflict(self, tool_name: str) -> str:
        """Generate new version if tool exists."""
        pass
```

---

## 3. Implementation

### 3.1 Promotion Flow

```python
def promote(self, tool_name: str) -> PromotionResult:
    """Promote tool to active registry."""
    
    try:
        # Check staging file exists
        staging_path = self.staging_dir / f"{tool_name}.py"
        if not staging_path.exists():
            return PromotionResult(
                success=False,
                error=f"Tool not found in staging: {tool_name}"
            )
        
        # Handle version conflict
        final_name = self._handle_version_conflict(tool_name)
        
        # Copy to active
        active_path = self._copy_tool_file(tool_name, final_name)
        
        # Update registry
        self._update_registry(final_name, active_path)
        
        # Clean up staging
        staging_path.unlink()
        
        return PromotionResult(
            success=True,
            active_path=active_path,
            final_name=final_name
        )
    
    except Exception as e:
        return PromotionResult(
            success=False,
            error=f"Promotion failed: {str(e)}"
        )


def _copy_tool_file(self, tool_name: str, final_name: str) -> Path:
    """Copy tool file."""
    import shutil
    
    src = self.staging_dir / f"{tool_name}.py"
    dst = self.active_dir / f"{final_name}.py"
    
    shutil.copy2(src, dst)
    return dst


def _update_registry(self, tool_name: str, tool_path: Path) -> None:
    """Update registry JSON."""
    
    # Load existing registry
    if self.registry_file.exists():
        with open(self.registry_file) as f:
            registry = json.load(f)
    else:
        registry = {"tools": []}
    
    # Add new tool
    registry["tools"].append({
        "name": tool_name,
        "path": str(tool_path),
        "promoted_at": datetime.now().isoformat(),
        "version": "1.0.0"
    })
    
    # Save
    with open(self.registry_file, "w") as f:
        json.dump(registry, f, indent=2)


def _handle_version_conflict(self, tool_name: str) -> str:
    """Check if tool exists, generate new version."""
    
    active_path = self.active_dir / f"{tool_name}.py"
    
    if not active_path.exists():
        return tool_name
    
    # Generate versioned name
    version = 2
    while (self.active_dir / f"{tool_name}_v{version}.py").exists():
        version += 1
    
    return f"{tool_name}_v{version}"
```

---

## 4. Testing

```python
def test_promote_new_tool():
    """Test promoting new tool."""
    staging_dir = Path("sandbox/staging")
    active_dir = Path("sandbox/active")
    registry_file = Path("sandbox/registry.json")
    
    # Setup
    staging_dir.mkdir(parents=True, exist_ok=True)
    active_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "test_tool.py").write_text("def test_tool(): pass")
    
    promoter = ToolPromoter(staging_dir, active_dir, registry_file)
    result = promoter.promote("test_tool")
    
    assert result.success
    assert (active_dir / "test_tool.py").exists()
    assert not (staging_dir / "test_tool.py").exists()


def test_version_conflict():
    """Test version handling."""
    # Create existing tool
    active_dir = Path("sandbox/active")
    active_dir.mkdir(parents=True, exist_ok=True)
    (active_dir / "test_tool.py").write_text("def test_tool(): pass")
    
    promoter = ToolPromoter(Path("staging"), active_dir, Path("registry.json"))
    new_name = promoter._handle_version_conflict("test_tool")
    
    assert new_name == "test_tool_v2"
```

---

**Estimated Lines of Code**: 300-400  
**Test Coverage Target**: >90%  
**Ready for Implementation**: ✅
