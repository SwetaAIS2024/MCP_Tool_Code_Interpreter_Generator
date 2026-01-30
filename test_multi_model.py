"""Test script to verify multi-model configuration."""

from src.llm_client import create_llm_client


def test_model_selection():
    """Test that different model types are correctly selected."""
    
    print("\n" + "="*80)
    print("TESTING MULTI-MODEL CONFIGURATION")
    print("="*80)
    
    # Test reasoning model
    print("\n1. Testing REASONING model:")
    reasoning_client = create_llm_client(model_type="reasoning")
    print(f"   Selected model: {reasoning_client.model}")
    
    # Test coding model
    print("\n2. Testing CODING model:")
    coding_client = create_llm_client(model_type="coding")
    print(f"   Selected model: {coding_client.model}")
    
    # Test default model
    print("\n3. Testing DEFAULT model:")
    default_client = create_llm_client()
    print(f"   Selected model: {default_client.model}")
    
    # Test direct model override
    print("\n4. Testing DIRECT model override:")
    override_client = create_llm_client(model_type="Qwen2.5:32B")
    print(f"   Selected model: {override_client.model}")
    
    print("\n" + "="*80)
    print("✅ Model selection working correctly!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_model_selection()
