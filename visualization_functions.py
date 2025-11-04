
import matplotlib.pyplot as plt

def plot_lane_polylines(scenario, figsize=(14, 12), show_lane_ids=False, show_lane_types=True):
    """
    Plot only lane polylines from a Waymo scenario.
    
    Args:
        scenario: Waymo scenario object
        figsize: Figure size tuple (width, height)
        show_lane_ids: Whether to show lane IDs on the plot
        show_lane_types: Whether to color-code lanes by type
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Lane type names and colors
    lane_type_names = {
        0: 'UNDEFINED',
        1: 'FREEWAY',
        2: 'SURFACE_STREET',
        3: 'BIKE_LANE'
    }
    
    lane_type_colors = {
        0: '#95a5a6',  # Gray for undefined
        1: '#e74c3c',  # Red for freeway
        2: '#3498db',  # Blue for surface street
        3: '#2ecc71'   # Green for bike lane
    }
    
    lane_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    print(f"Plotting lane polylines for scenario: {scenario.scenario_id}")
    
    # Plot lane polylines
    for feature in scenario.map_features:
        feature_type = feature.WhichOneof('feature_data')
        
        if feature_type == 'lane' and hasattr(feature.lane, 'polyline'):
            lane_type = feature.lane.type
            lane_counts[lane_type] += 1
            
            # Extract coordinates
            x_coords = [point.x for point in feature.lane.polyline]
            y_coords = [point.y for point in feature.lane.polyline]
            
            # Choose color
            if show_lane_types:
                color = lane_type_colors.get(lane_type, '#95a5a6')
            else:
                color = '#3498db'  # Default blue
            
            # Plot the lane
            ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
            
            # Optionally show lane ID at the midpoint
            if show_lane_ids and len(x_coords) > 0:
                mid_idx = len(x_coords) // 2
                ax.annotate(
                    str(feature.id),
                    xy=(x_coords[mid_idx], y_coords[mid_idx]),
                    fontsize=8,
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )
    
    # Configure plot
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Lane Polylines - Scenario: {scenario.scenario_id}', 
                 fontsize=14, fontweight='bold')
    
    # Create legend if showing lane types
    if show_lane_types:
        for lane_type, count in lane_counts.items():
            if count > 0:
                type_name = lane_type_names.get(lane_type, f'Type_{lane_type}')
                color = lane_type_colors.get(lane_type, '#95a5a6')
                ax.plot([], [], color=color, linewidth=2, alpha=0.7, 
                       label=f'{type_name} ({count})')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Print summary
    total_lanes = sum(lane_counts.values())
    print(f"Total lanes plotted: {total_lanes}")
    for lane_type, count in lane_counts.items():
        if count > 0:
            type_name = lane_type_names.get(lane_type, f'Type_{lane_type}')
            print(f"  - {type_name}: {count}")
    
    return fig, ax


# Example usage:
fig, ax = plot_lane_polylines(
    training_dataset[0][0],
    figsize=(14, 12),
    show_lane_ids=False,
    show_lane_types=True
)




# VISUALIZE SCENARIO WITH GRAPH EDGES
import matplotlib.pyplot as plt

def show_distances(pos, src_idx, dst_idx, x_coords, y_coords):
    dist = np.sqrt((pos[src_idx, 0] - pos[dst_idx, 0])**2 + (pos[src_idx, 1] - pos[dst_idx, 1])**2)
    mid_x = (x_coords[0] + x_coords[1]) / 2
    mid_y = (y_coords[0] + y_coords[1]) / 2
    ax.annotate(f'{dist:.1f}m', xy=(mid_x, mid_y), fontsize=6, color='blue', ha='center', alpha=0.6)

def visualize_scenario_with_graph(scenario, timestep, radius, figsize=(14, 12), show_future=True, show_edge_distances=False, show_scenario_analysis=False):

    if show_scenario_analysis:
        analyze_scenario(scenario)
        analyze_scenario_agents(scenario)

    graph_data = scenario_to_pyg_data(scenario, timestep, radius, future_states=10, method='star')
    if graph_data is None:
        print("No valid agents at this timestep!")
        return None, None, None

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = {1: '#e74c3c', 2: '#3498db', 3: '#f39c12', 4: '#9b59b6'}
    type_names = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist', 4: 'Other'}
    print(f"VISUALIZING SCENARIO WITH GRAPH: {scenario.scenario_id}")
    print(f" Time: {scenario.timestamps_seconds[timestep]:.1f}s")
    print(f" Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")

    render_map_features(scenario, ax)

    print(f" Rendering {graph_data.num_edges} graph edges...")
    edge_index = graph_data.edge_index
    pos = graph_data.pos.numpy()
    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i].item()
        dst_idx = edge_index[1, i].item()
        x_coords = [pos[src_idx, 0], pos[dst_idx, 0]]
        y_coords = [pos[src_idx, 1], pos[dst_idx, 1]]
        ax.plot(x_coords, y_coords, 'blue', linewidth=1.2, alpha=0.4, zorder=5)
        if show_edge_distances:
            show_distances()

    render_agents(scenario, timestep, colors, ax, show_future=show_future)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Waymo Scenario with Graph Edges (radius={radius}m)\n'
                f'Scenario: {scenario.scenario_id}, Time: {scenario.timestamps_seconds[timestep]:.1f}s\n'
                f'{graph_data.num_nodes} nodes, {graph_data.num_edges} edges', 
                fontsize=14, fontweight='bold')
    all_x = [state.center_x for track in scenario.tracks for state in track.states if state.valid]
    all_y = [state.center_y for track in scenario.tracks for state in track.states if state.valid]
    if all_x and all_y:
        margin = 20
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    for obj_type, color in colors.items():
        if obj_type in [t.object_type for t in scenario.tracks]:
            ax.scatter([], [], c=color, s=50, label=type_names[obj_type])
    ax.plot([], [], 'blue', linewidth=2, alpha=0.4, label=f'Graph edges (≤{radius}m)')
    ax.legend()
    plt.tight_layout()

    agent_counts = {}
    for track in scenario.tracks:
        if timestep < len(track.states) and track.states[timestep].valid:
            agent_type = type_names.get(track.object_type, 'Other')
            agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
    print(" Agents in graph:")
    for agent_type, count in agent_counts.items():
        print(f"   - {count} {agent_type}{'s' if count != 1 else ''}")
    if graph_data.num_edges > 0:
        avg_degree = graph_data.num_edges / graph_data.num_nodes
        print(f" Average degree: {avg_degree:.2f} edges per node")
    
    return fig, ax, graph_data

# %%
# TEST: Visualize scenario with graph edges at different radii
time = 0
fig1, ax1, graph1 = visualize_scenario_with_graph(training_dataset[0][0], timestep=time, radius=30.0, show_future=True)
