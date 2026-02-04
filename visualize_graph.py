"""Visualize the LangGraph pipeline as a graph image."""

from src.pipeline import build_graph
from pathlib import Path

def visualize_pipeline():
    """Generate and save pipeline graph visualization."""
    
    # Build the graph
    graph = build_graph()
    
    # Get the graph structure
    graph_structure = graph.get_graph()
    
    # Generate Mermaid diagram (text-based)
    print("\n" + "="*80)
    print("PIPELINE GRAPH (Mermaid Format)")
    print("="*80)
    mermaid = graph_structure.draw_mermaid()
    print(mermaid)
    
    # Save Mermaid to file
    mermaid_file = Path("docs/pipeline_graph.mmd")
    mermaid_file.parent.mkdir(parents=True, exist_ok=True)
    mermaid_file.write_text(mermaid)
    print(f"\n✅ Mermaid diagram saved to: {mermaid_file}")
    
    # Try to generate PNG if graphviz is available
    try:
        png_data = graph_structure.draw_mermaid_png()
        png_file = Path("docs/pipeline_graph.png")
        png_file.write_bytes(png_data)
        print(f"✅ PNG diagram saved to: {png_file}")
    except Exception as e:
        print(f"\n⚠️  Could not generate PNG (graphviz may not be installed): {e}")
        print("   You can paste the Mermaid code into https://mermaid.live to visualize")
    
    # Print ASCII representation
    print("\n" + "="*80)
    print("PIPELINE GRAPH (ASCII)")
    print("="*80)
    try:
        ascii_graph = graph_structure.draw_ascii()
        print(ascii_graph)
    except Exception as e:
        print(f"ASCII visualization not available: {e}")
    
    print("\n" + "="*80)
    print("GRAPH NODES:")
    print("="*80)
    for node in graph_structure.nodes:
        print(f"  • {node}")
    
    print("\n" + "="*80)
    print("GRAPH EDGES:")
    print("="*80)
    for edge in graph_structure.edges:
        print(f"  • {edge}")

if __name__ == "__main__":
    visualize_pipeline()
