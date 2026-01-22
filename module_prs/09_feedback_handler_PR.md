# Module PR 09: Feedback Handler

**Module**: `src/feedback_handler.py`  
**Priority**: P0 (Critical decision point)  
**Estimated Effort**: 1-2 days  
**Dependencies**: `01_data_models`

---

## 1. Module Purpose

The Feedback Handler parses user responses and makes approval decisions:
- **Stage 1**: "Yes" → proceed to stage 2, "No" → reject
- **Stage 2**: "Approve" → promote tool, "Reject" → archive

**Key Principle**: Strict token matching. No fuzzy parsing. Ambiguous responses are treated as rejection.

---

## 2. Core Components

```python
class FeedbackHandler:
    """Parse user feedback and make decisions."""
    
    def parse_stage1_response(self, response: str) -> bool:
        """
        Parse stage 1 response (output validation).
        
        Args:
            response: User input
        
        Returns:
            True if output is correct, False otherwise
        """
        pass
    
    def parse_stage2_response(self, response: str) -> UserFeedback:
        """
        Parse stage 2 response (registration approval).
        
        Args:
            response: User input
        
        Returns:
            UserFeedback with decision and reason
        """
        pass
    
    def _normalize_response(self, response: str) -> str:
        """Clean and normalize user input."""
        pass
```

---

## 3. Implementation

### 3.1 Stage 1 Parsing

```python
def parse_stage1_response(self, response: str) -> bool:
    """
    Parse output validation response.
    
    Accepted: "Yes", "yes", "YES"
    Rejected: Everything else
    """
    normalized = self._normalize_response(response)
    
    return normalized in ["yes", "y"]


def _normalize_response(self, response: str) -> str:
    """Normalize response."""
    return response.strip().lower()
```

### 3.2 Stage 2 Parsing

```python
def parse_stage2_response(self, response: str) -> UserFeedback:
    """
    Parse registration approval response.
    
    Approved: "Approve", "approve", "APPROVE"
    Rejected: "Reject", "reject", "REJECT", or anything else
    """
    normalized = self._normalize_response(response)
    
    if normalized == "approve":
        return UserFeedback(
            decision="APPROVED",
            reason="User approved tool registration",
            timestamp=datetime.now()
        )
    
    elif normalized == "reject":
        return UserFeedback(
            decision="REJECTED",
            reason="User rejected tool registration",
            timestamp=datetime.now()
        )
    
    else:
        # Ambiguous → reject for safety
        return UserFeedback(
            decision="REJECTED",
            reason=f"Ambiguous response: '{response}'. Expected 'Approve' or 'Reject'.",
            timestamp=datetime.now()
        )
```

---

## 4. Testing

```python
def test_stage1_yes_responses():
    """Test stage 1 acceptance."""
    handler = FeedbackHandler()
    
    assert handler.parse_stage1_response("Yes") == True
    assert handler.parse_stage1_response("yes") == True
    assert handler.parse_stage1_response("y") == True


def test_stage1_rejection():
    """Test stage 1 rejection."""
    handler = FeedbackHandler()
    
    assert handler.parse_stage1_response("No") == False
    assert handler.parse_stage1_response("Maybe") == False
    assert handler.parse_stage1_response("") == False


def test_stage2_approve():
    """Test stage 2 approval."""
    handler = FeedbackHandler()
    
    feedback = handler.parse_stage2_response("Approve")
    assert feedback.decision == "APPROVED"


def test_stage2_reject():
    """Test stage 2 rejection."""
    handler = FeedbackHandler()
    
    feedback = handler.parse_stage2_response("Reject")
    assert feedback.decision == "REJECTED"


def test_stage2_ambiguous():
    """Test ambiguous responses default to reject."""
    handler = FeedbackHandler()
    
    feedback = handler.parse_stage2_response("ok")
    assert feedback.decision == "REJECTED"
    assert "Ambiguous" in feedback.reason
```

---

**Estimated Lines of Code**: 150-200  
**Test Coverage Target**: >95%  
**Ready for Implementation**: ✅
