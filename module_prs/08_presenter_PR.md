# Module PR 08: Presenter

**Module**: `src/presenter.py`  
**Priority**: P1 (User interface)  
**Estimated Effort**: 1-2 days  
**Dependencies**: `01_data_models`, `07_executor`

---

## 1. Module Purpose

The Presenter formats tool execution results and requests user approval with:
- **Output Display** - Formatted DataFrame, summary, metadata
- **Two-Stage Approval** - (1) Output validation, (2) Registration decision
- **Clear Instructions** - Exact tokens required (Approve/Reject)

**Key Principle**: Make approval unambiguous. User must explicitly type "Approve" or "Reject" (case-sensitive).

---

## 2. Core Components

```python
class ResultPresenter:
    """Format and present tool execution results."""
    
    def present_results(
        self,
        tool_name: str,
        artifacts: RunArtifacts,
        spec: ToolSpec
    ) -> str:
        """
        Format results as markdown for display.
        
        Args:
            tool_name: Name of executed tool
            artifacts: Execution results
            spec: Tool specification
        
        Returns:
            Formatted markdown string
        """
        pass
    
    def generate_approval_prompt(
        self,
        tool_name: str,
        stage: int  # 1 or 2
    ) -> str:
        """
        Generate approval request prompt.
        
        Stage 1: "Is the output correct?"
        Stage 2: "Approve tool for registration?"
        """
        pass
    
    def _format_dataframe_preview(self, df_dict: Dict) -> str:
        """Format DataFrame as markdown table."""
        pass
    
    def _format_metadata(self, metadata: Dict) -> str:
        """Format execution metadata."""
        pass
```

---

## 3. Implementation

### 3.1 Result Presentation

```python
def present_results(
    self,
    tool_name: str,
    artifacts: RunArtifacts,
    spec: ToolSpec
) -> str:
    """Format execution results as markdown."""
    
    if artifacts.error:
        return f"""
## ❌ Tool Execution Failed: {tool_name}

**Error**: {artifacts.error}
**Execution Time**: {artifacts.execution_time:.2f}s

The tool encountered an error during execution.
"""
    
    # Format result
    result_md = self._format_result(artifacts.result)
    
    # Format metadata
    metadata_md = f"""
**Execution Time**: {artifacts.execution_time:.2f}s
**Memory Used**: {artifacts.memory_used:.2f} MB
**Rows**: {artifacts.result.get('metadata', {}).get('row_count', 'N/A')}
**Columns**: {artifacts.result.get('metadata', {}).get('columns', [])}
"""
    
    return f"""
## ✓ Tool Executed Successfully: {tool_name}

### Description
{spec.description}

### Results
{result_md}

### Summary
{artifacts.summary_markdown or 'No summary provided'}

### Execution Metrics
{metadata_md}
"""


def _format_result(self, result: Dict) -> str:
    """Format result data as markdown."""
    
    if "result" in result and isinstance(result["result"], list):
        # DataFrame result
        df = pd.DataFrame(result["result"])
        
        if len(df) <= 20:
            # Show full table
            return df.to_markdown(index=False)
        else:
            # Show preview
            preview = df.head(10)
            return f"{preview.to_markdown(index=False)}\n\n... ({len(df) - 10} more rows)"
    
    else:
        # Other result types
        return f"```json\n{json.dumps(result, indent=2)}\n```"
```

### 3.2 Approval Prompts

```python
def generate_approval_prompt(self, tool_name: str, stage: int) -> str:
    """Generate two-stage approval prompt."""
    
    if stage == 1:
        return f"""
---

## Stage 1: Output Validation

Is the output from `{tool_name}` **correct and as expected**?

**Instructions**:
- If the results look correct, respond: `Yes`
- If there are issues, respond: `No` (tool will be rejected)

Your response: """
    
    elif stage == 2:
        return f"""
---

## Stage 2: Registration Approval

The output is correct. Do you want to **register `{tool_name}` for future use**?

**Instructions**:
- To register the tool, respond: `Approve`
- To reject registration, respond: `Reject`

**Note**: This action will move the tool to the active registry.

Your response: """
    
    else:
        raise ValueError(f"Invalid stage: {stage}")
```

---

## 4. Testing

```python
def test_present_successful_results():
    """Test result formatting."""
    artifacts = RunArtifacts(
        result={
            "result": [{"state": "CA", "count": 100}, {"state": "TX", "count": 80}],
            "metadata": {"row_count": 2, "columns": ["state", "count"]}
        },
        summary_markdown="Grouped by state",
        execution_time=1.23,
        memory_used=50.0
    )
    
    spec = ToolSpec(
        tool_name="group_by_state",
        description="Group by state",
        input_schema={},
        output_schema={},
        constraints=[]
    )
    
    presenter = ResultPresenter()
    output = presenter.present_results("group_by_state", artifacts, spec)
    
    assert "✓ Tool Executed Successfully" in output
    assert "1.23s" in output
    assert "Grouped by state" in output
```

---

## 5. Configuration

```yaml
presentation:
  max_table_rows: 20
  format: "markdown"
  show_execution_metrics: true
```

---

**Estimated Lines of Code**: 250-350  
**Test Coverage Target**: >85%  
**Ready for Implementation**: ✅
