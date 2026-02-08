#!/usr/bin/env python3
"""
Test script for Program Specification functionality.

This script demonstrates:
1. Creating a ProgramSpec manually (without utility functions)
2. Validating the program specification
3. Converting ProgramSpec to GraphSpec
4. Validating the generated graph
5. Generating mermaid diagram and image visualization
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


import asyncio
import json
from typing import Dict, List, Any

# Import our program and graph models
from kgraphplanner.program.program import (
    ProgramSpec, StaticNodeSpec, StaticEdgeSpec, TemplateSpec,
    ForEachSpec, NodePattern, EdgePattern, FanOutSpec, FanInSpec,
    FanOutBranch, BindingPattern, NodeDefaults
)
from kgraphplanner.graph.exec_graph import (
    GraphSpec, validate_graph_spec, WorkerNodeSpec, StartNodeSpec, 
    EndNodeSpec, EdgeSpec, Binding
)

def create_test_program() -> ProgramSpec:
    """Create a test program specification manually without utility functions."""
    
    # Define companies to analyze
    companies = ["OpenAI", "Anthropic", "Google DeepMind", "Cohere", "Mistral"]
    
    return ProgramSpec(
        program_id="ai_company_analysis",
        name="AI Company Analysis Program",
        description="Analyze multiple AI companies using parallel research and dual analysis tracks",
        version="1.0",
        
        # Static nodes that don't change
        static_nodes=[
            StaticNodeSpec(
                id="start",
                worker="start",
                defaults=NodeDefaults(
                    args={"companies": companies, "analysis_depth": "comprehensive"}
                )
            ),
            StaticNodeSpec(
                id="final_aggregator",
                worker="report_aggregator",
                defaults=NodeDefaults(
                    prompt="Compile all company analyses into a comprehensive market report"
                )
            ),
            StaticNodeSpec(
                id="end",
                worker="end"
            )
        ],
        
        # Static edges that don't change
        static_edges=[
            StaticEdgeSpec(
                source="final_aggregator",
                destination="end",
                prompt="Finalize the comprehensive AI market analysis report"
            )
        ],
        
        # Dynamic templates that expand based on data
        templates=[
            TemplateSpec(
                name="company_analysis_pipeline",
                description="For each company, research and analyze through two different analytical approaches",
                
                # Iterate over companies
                for_each=ForEachSpec(
                    source_from="start_args",
                    source_path="$.companies",
                    item_var="company",
                    idx_var="company_idx"
                ),
                
                # Create nodes for each company
                loop_nodes=[
                    NodePattern(
                        id_tpl="research_{company_idx}",
                        worker="research_worker",
                        defaults_tpl=NodeDefaults(
                            prompt="Research the company: {company}",
                            args={"target_company": "{company}", "research_type": "comprehensive"}
                        )
                    ),
                    NodePattern(
                        id_tpl="financial_analysis_{company_idx}",
                        worker="financial_analyst",
                        defaults_tpl=NodeDefaults(
                            prompt="Perform financial analysis for {company}",
                            args={"company_name": "{company}", "analysis_focus": "financial"}
                        )
                    ),
                    NodePattern(
                        id_tpl="technical_analysis_{company_idx}",
                        worker="technical_analyst", 
                        defaults_tpl=NodeDefaults(
                            prompt="Perform technical analysis for {company}",
                            args={"company_name": "{company}", "analysis_focus": "technical"}
                        )
                    ),
                    NodePattern(
                        id_tpl="synthesis_{company_idx}",
                        worker="synthesis_analyst",
                        defaults_tpl=NodeDefaults(
                            prompt="Synthesize financial and technical analyses for {company}",
                            args={"company_name": "{company}"}
                        )
                    )
                ],
                
                # Create edges between nodes in the template
                loop_edges=[
                    EdgePattern(
                        source_tpl="start",
                        destination_tpl="research_{company_idx}",
                        prompt_tpl="Begin research for {company}",
                        bindings={
                            "company_info": [BindingPattern(
                                from_node_tpl="start",
                                path="$.companies[{company_idx}]",
                                alias="target_company"
                            )]
                        }
                    )
                ],
                
                # Fan-out from research to both analysis tracks
                fan_out=[
                    FanOutSpec(
                        source_tpl="research_{company_idx}",
                        description="Split research results to financial and technical analysis",
                        branches=[
                            FanOutBranch(
                                destination_tpl="financial_analysis_{company_idx}",
                                prompt_tpl="Analyze financial aspects of {company}",
                                bindings={
                                    "research_data": [BindingPattern(
                                        from_node_tpl="research_{company_idx}",
                                        path="$",
                                        transform="json"
                                    )]
                                }
                            ),
                            FanOutBranch(
                                destination_tpl="technical_analysis_{company_idx}",
                                prompt_tpl="Analyze technical aspects of {company}",
                                bindings={
                                    "research_data": [BindingPattern(
                                        from_node_tpl="research_{company_idx}",
                                        path="$",
                                        transform="json"
                                    )]
                                }
                            )
                        ]
                    )
                ],
                
                # Fan-in from both analysis tracks to synthesis
                fan_in=[
                    FanInSpec(
                        sources_tpl=["financial_analysis_{company_idx}", "technical_analysis_{company_idx}"],
                        destination_tpl="synthesis_{company_idx}",
                        param="analysis_results",
                        reduce="merge_dict",
                        prompt_tpl="Combine financial and technical analyses for {company}",
                        description="Merge both analysis tracks for comprehensive view"
                    ),
                    # Aggregate all company syntheses to final report
                    FanInSpec(
                        sources_tpl=["synthesis_{company_idx}"],
                        destination_tpl="final_aggregator",
                        param="company_analyses",
                        reduce="append_list",
                        prompt_tpl="Include {company} analysis in final report",
                        description="Collect all company analyses for final aggregation"
                    )
                ]
            )
        ],
        
        # Configuration
        exit_nodes=["end"],
        max_parallel=3,
        global_context={"analysis_date": "2024-01-01", "market_focus": "AI/ML"},
        required_workers=["research_worker", "financial_analyst", "technical_analyst", "synthesis_analyst", "report_aggregator"]
    )

def validate_program_spec(program: ProgramSpec) -> bool:
    """Validate the program specification for correctness."""
    print("🔍 Validating Program Specification...")
    
    # Check basic structure
    if not program.program_id:
        print("❌ Program ID is required")
        return False
    
    if not program.static_nodes:
        print("❌ At least one static node is required")
        return False
    
    # Check for start and end nodes
    node_ids = [node.id for node in program.static_nodes]
    if "start" not in node_ids:
        print("❌ 'start' node is required")
        return False
    
    if "end" not in node_ids:
        print("❌ 'end' node is required")
        return False
    
    # Validate templates
    for template in program.templates:
        if not template.name:
            print(f"❌ Template missing name")
            return False
        
        if not template.for_each:
            print(f"❌ Template '{template.name}' missing for_each specification")
            return False
        
        # Check that we have some nodes or patterns to expand
        if not (template.loop_nodes or template.fan_out or template.fan_in):
            print(f"❌ Template '{template.name}' has no expandable content")
            return False
    
    # Check required workers are specified
    if program.required_workers:
        template_workers = set()
        for template in program.templates:
            for node_pattern in template.loop_nodes:
                template_workers.add(node_pattern.worker)
        
        static_workers = {node.worker for node in program.static_nodes}
        all_workers = template_workers | static_workers
        
        missing_workers = set(program.required_workers) - all_workers
        if missing_workers:
            print(f"⚠️ Warning: Required workers not found in program: {missing_workers}")
    
    print("✅ Program specification is valid")
    return True

def expand_program_to_graph(program: ProgramSpec) -> GraphSpec:
    """Convert ProgramSpec to GraphSpec by expanding templates."""
    print("🔄 Converting Program to Graph...")
    
    # Start with static nodes converted to WorkerNodeSpec format
    nodes = []
    for static_node in program.static_nodes:
        if static_node.id == "start":
            nodes.append(StartNodeSpec(
                id=static_node.id,
                initial_data=static_node.defaults.model_dump() if static_node.defaults else {}
            ))
        elif static_node.id == "end":
            nodes.append(EndNodeSpec(id=static_node.id))
        else:
            nodes.append(WorkerNodeSpec(
                id=static_node.id,
                worker_name=static_node.worker,
                defaults=static_node.defaults.model_dump() if static_node.defaults else {}
            ))
    
    # Convert static edges
    edges = []
    for static_edge in program.static_edges:
        edges.append(EdgeSpec(
            source=static_edge.source,
            destination=static_edge.destination,
            prompt=static_edge.prompt
        ))
    
    # Get start args for template expansion
    start_node = program.get_static_node_by_id("start")
    start_args = {}
    if start_node and start_node.defaults and start_node.defaults.args:
        start_args = start_node.defaults.args
    
    # Expand templates
    for template in program.templates:
        print(f"  📋 Expanding template: {template.name}")
        
        # Get items to iterate over
        if template.for_each.source_from == "literal":
            items = template.for_each.literal_items or []
        else:
            # Simple path resolution for $.companies
            path = template.for_each.source_path
            if path.startswith("$."):
                key = path[2:]
                items = start_args.get(key, [])
            else:
                items = []
        
        print(f"    🔢 Processing {len(items)} items: {items}")
        
        # Expand for each item
        for idx, item in enumerate(items):
            context = {template.for_each.item_var: item, template.for_each.idx_var: idx}
            
            # Expand loop nodes
            for node_pattern in template.loop_nodes:
                node_id = node_pattern.id_tpl.format(**context)
                
                defaults = {}
                if node_pattern.defaults_tpl:
                    # Format the defaults with context
                    defaults_dict = node_pattern.defaults_tpl.model_dump()
                    if defaults_dict.get("args"):
                        formatted_args = {}
                        for k, v in defaults_dict["args"].items():
                            if isinstance(v, str):
                                formatted_args[k] = v.format(**context)
                            else:
                                formatted_args[k] = v
                        defaults["args"] = formatted_args
                    if defaults_dict.get("prompt"):
                        defaults["prompt"] = defaults_dict["prompt"].format(**context)
                
                nodes.append(WorkerNodeSpec(
                    id=node_id,
                    worker_name=node_pattern.worker,
                    defaults=defaults
                ))
            
            # Expand loop edges
            for edge_pattern in template.loop_edges:
                source = edge_pattern.source_tpl.format(**context)
                destination = edge_pattern.destination_tpl.format(**context)
                prompt = edge_pattern.prompt_tpl.format(**context) if edge_pattern.prompt_tpl else None
                
                # Convert binding patterns to bindings
                bindings = {}
                for param, binding_patterns in edge_pattern.bindings.items():
                    binding_list = []
                    for bp in binding_patterns:
                        from_node = bp.from_node_tpl.format(**context) if bp.from_node_tpl else None
                        path = bp.path.format(**context) if bp.path else "$"
                        binding_list.append(Binding(
                            from_node=from_node,
                            path=path,
                            transform=bp.transform,
                            reduce=bp.reduce,
                            alias=bp.alias
                        ))
                    bindings[param] = binding_list
                
                edges.append(EdgeSpec(
                    source=source,
                    destination=destination,
                    prompt=prompt,
                    bindings=bindings
                ))
            
            # Expand fan-out
            for fan_out in template.fan_out:
                source = fan_out.source_tpl.format(**context)
                for branch in fan_out.branches:
                    destination = branch.destination_tpl.format(**context)
                    prompt = branch.prompt_tpl.format(**context) if branch.prompt_tpl else None
                    
                    # Convert bindings
                    bindings = {}
                    for param, binding_patterns in branch.bindings.items():
                        binding_list = []
                        for bp in binding_patterns:
                            from_node = bp.from_node_tpl.format(**context) if bp.from_node_tpl else None
                            binding_list.append(Binding(
                                from_node=from_node,
                                path=bp.path,
                                transform=bp.transform,
                                reduce=bp.reduce,
                                alias=bp.alias
                            ))
                        bindings[param] = binding_list
                    
                    edges.append(EdgeSpec(
                        source=source,
                        destination=destination,
                        prompt=prompt,
                        bindings=bindings
                    ))
            
            # Expand fan-in
            for fan_in in template.fan_in:
                destination = fan_in.destination_tpl.format(**context)
                prompt = fan_in.prompt_tpl.format(**context) if fan_in.prompt_tpl else None
                
                for source_tpl in fan_in.sources_tpl:
                    source = source_tpl.format(**context)
                    
                    bindings = {fan_in.param: [Binding(
                        from_node=source,
                        path="$",
                        reduce=fan_in.reduce
                    )]}
                    
                    edges.append(EdgeSpec(
                        source=source,
                        destination=destination,
                        prompt=prompt,
                        bindings=bindings
                    ))
    
    # Create the graph spec
    graph_spec = GraphSpec(
        graph_id=program.program_id,
        name=program.name,
        description=program.description,
        version=program.version,
        nodes=nodes,
        edges=edges,
        exit_points=program.exit_nodes,
        max_parallel_execution=program.max_parallel
    )
    
    print(f"✅ Generated graph with {len(nodes)} nodes and {len(edges)} edges")
    return graph_spec

def create_mermaid_diagram(graph_spec: GraphSpec) -> str:
    """Create a mermaid diagram from the graph specification."""
    print("📊 Generating Mermaid diagram...")
    
    lines = ["graph TD"]
    
    # Reserved keywords in Mermaid that need to be escaped
    reserved_keywords = {'end', 'start', 'graph', 'subgraph', 'style', 'class', 'click', 'direction'}
    
    def safe_node_id(node_id: str) -> str:
        """Make node ID safe for Mermaid by escaping reserved keywords."""
        if node_id.lower() in reserved_keywords:
            return f"node_{node_id}"
        return node_id
    
    # Add nodes
    for node in graph_spec.nodes:
        safe_id = safe_node_id(node.id)
        node_label = f"{node.id}"
        if hasattr(node, 'worker_name'):
            node_label += f" ({node.worker_name})"
        elif hasattr(node, 'node_type'):
            node_label += f" [{node.node_type}]"
        
        # Different shapes for different node types
        if hasattr(node, 'node_type'):
            if node.node_type == "start":
                lines.append(f'    {safe_id}(("{node_label}"))')
            elif node.node_type == "end":
                lines.append(f'    {safe_id}["{node_label}"]')
            else:
                lines.append(f'    {safe_id}["{node_label}"]')
        else:
            lines.append(f'    {safe_id}["{node_label}"]')
    
    # Add edges
    for edge in graph_spec.edges:
        safe_source = safe_node_id(edge.source)
        safe_dest = safe_node_id(edge.destination)
        
        edge_label = ""
        if edge.prompt:
            # Truncate long prompts and escape quotes
            prompt = edge.prompt[:30] + "..." if len(edge.prompt) > 30 else edge.prompt
            # Escape quotes and newlines for Mermaid
            prompt = prompt.replace('"', "'").replace('\n', ' ')
            edge_label = f'|"{prompt}"|'
        
        lines.append(f'    {safe_source} -->{edge_label} {safe_dest}')
    
    mermaid_syntax = "\n".join(lines)
    print("✅ Mermaid diagram generated")
    return mermaid_syntax

async def generate_graph_image(mermaid_syntax: str, output_filename: str = "program_graph.png"):
    """Generate a PNG image from mermaid syntax."""
    print(f"🖼️ Generating graph image: {output_filename}")
    
    try:
        from langchain_core.runnables.graph_mermaid import _render_mermaid_using_pyppeteer
        
        image_bytes = await _render_mermaid_using_pyppeteer(
            mermaid_syntax,
            output_file_path=None,
            background_color="white",
            padding=10
        )
        
        with open(output_filename, "wb") as f:
            f.write(image_bytes)
        
        print(f"✅ Graph image saved as {output_filename}")
        return True
        
    except Exception as e:
        print(f"❌ Could not generate graph image: {e}")
        print("📊 Continuing without image generation...")
        return False

async def main():
    """Main test function."""
    print("🚀 Testing Program Specification System")
    print("=" * 50)
    
    # Step 1: Create program specification
    print("\n1️⃣ Creating Program Specification")
    program = create_test_program()
    print(f"✅ Created program: {program.name}")
    print(f"   📋 Templates: {len(program.templates)}")
    print(f"   🏗️ Static nodes: {len(program.static_nodes)}")
    print(f"   🔗 Static edges: {len(program.static_edges)}")
    
    # Step 2: Validate program
    print("\n2️⃣ Validating Program Specification")
    if not validate_program_spec(program):
        print("❌ Program validation failed!")
        return
    
    # Step 3: Convert to graph
    print("\n3️⃣ Converting Program to Graph")
    try:
        graph_spec = expand_program_to_graph(program)
    except Exception as e:
        print(f"❌ Failed to convert program to graph: {e}")
        return
    
    # Step 4: Validate graph
    print("\n4️⃣ Validating Generated Graph")
    validation_result = validate_graph_spec(graph_spec)
    
    if validation_result.has_errors():
        print("❌ Graph validation failed!")
        for error in validation_result.errors:
            print(f"   🔴 {error.message}")
        return
    
    if validation_result.has_warnings():
        print("⚠️ Graph validation warnings:")
        for warning in validation_result.warnings:
            print(f"   🟡 {warning.message}")
    else:
        print("✅ Graph validation passed!")
    
    # Step 5: Generate mermaid diagram
    print("\n5️⃣ Generating Mermaid Diagram")
    mermaid_syntax = create_mermaid_diagram(graph_spec)
    
    # Save mermaid syntax to file
    with open("program_graph.mmd", "w") as f:
        f.write(mermaid_syntax)
    print("✅ Mermaid syntax saved to program_graph.mmd")
    
    # Step 6: Generate image
    print("\n6️⃣ Generating Graph Image")
    await generate_graph_image(mermaid_syntax)
    
    # Step 7: Summary
    print("\n📊 Summary")
    print("=" * 50)
    print(f"Program ID: {graph_spec.graph_id}")
    print(f"Nodes: {len(graph_spec.nodes)}")
    print(f"Edges: {len(graph_spec.edges)}")
    print(f"Exit points: {graph_spec.exit_points}")
    print(f"Max parallel: {graph_spec.max_parallel_execution}")
    
    # Show node breakdown
    node_types = {}
    for node in graph_spec.nodes:
        node_type = getattr(node, 'node_type', 'worker')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\nNode breakdown:")
    for node_type, count in node_types.items():
        print(f"  {node_type}: {count}")
    
    print("\n🎉 Program specification test completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
