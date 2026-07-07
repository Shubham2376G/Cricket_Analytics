import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from IPython.display import HTML


# Load pose estimation JSON
def load_pose_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


# Define skeleton connections
SKELETON = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
            [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
            [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]


# Extract frames from JSON data
def extract_frames(data):
    frames = []
    for frame in data["frames"]:
        joints = np.array([[j["x"], j["y"], j["z"]] for j in frame["joints"]])
        frames.append(joints)
    return frames


# Create cricket pitch elements
def create_cricket_pitch():
    # Cricket pitch dimensions (scaled to fit coordinate system)
    pitch_length = 6.8  # Length along y-axis
    pitch_width = 1.5  # Width along x-axis

    # Define the corners of the pitch
    x = [-pitch_width / 2, pitch_width / 2, pitch_width / 2, -pitch_width / 2, -pitch_width / 2]
    y = [0, 0, pitch_length, pitch_length, 0]
    z = [0, 0, 0, 0, 0]

    # Create pitch surface
    pitch_surface = go.Mesh3d(
        x=[-pitch_width / 2, pitch_width / 2, pitch_width / 2, -pitch_width / 2],
        y=[0, 0, pitch_length, pitch_length],
        z=[0, 0, 0, 0],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color='tan',
        opacity=0.7,
        name='Cricket Pitch'
    )

    # Create pitch outline
    pitch_outline = go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color='brown', width=2),
        name='Pitch Outline'
    )

    # Bowling crease at one end
    bowling_crease1 = go.Scatter3d(
        x=[-pitch_width / 2 - 0.05, pitch_width / 2 + 0.05],
        y=[0 + 0.2,0 + 0.2],
        z=[0.001, 0.001],
        mode='lines',
        line=dict(color='white', width=4),
        name='Bowling Crease 1'
    )

    # Bowling crease at other end
    bowling_crease2 = go.Scatter3d(
        x=[-pitch_width / 2 - 0.05, pitch_width / 2 + 0.05],
        y=[pitch_length- 0.2, pitch_length - 0.2],
        z=[0.001, 0.001],
        mode='lines',
        line=dict(color='white', width=4),
        name='Bowling Crease 2'
    )

    # Popping creases
    popping_creases = []

    # Near end
    popping_creases.append(go.Scatter3d(
        x=[-pitch_width / 2, -pitch_width / 2],
        y=[- 0.05, 0.15],
        z=[0.001, 0.001],
        mode='lines',
        line=dict(color='white', width=4),
        name='Popping Crease 1',
        showlegend=False
    ))

    popping_creases.append(go.Scatter3d(
        x=[pitch_width / 2, pitch_width / 2],
        y=[- 0.05, 0.15],
        z=[0.001, 0.001],
        mode='lines',
        line=dict(color='white', width=4),
        name='Popping Crease 2',
        showlegend=False
    ))

    # Far end
    popping_creases.append(go.Scatter3d(
        x=[-pitch_width / 2, -pitch_width / 2],
        y=[pitch_length + 0.05, pitch_length - 0.15],
        z=[0.001, 0.001],
        mode='lines',
        line=dict(color='white', width=4),
        name='Popping Crease 3',
        showlegend=False
    ))

    popping_creases.append(go.Scatter3d(
        x=[pitch_width / 2, pitch_width / 2],
        y=[pitch_length + 0.05, pitch_length - 0.15],
        z=[0.001, 0.001],
        mode='lines',
        line=dict(color='white', width=4),
        name='Popping Crease 4',
        showlegend=False
    ))

    # Create field surface
    field_size = 1.2
    field_surface = go.Mesh3d(
        x=[-field_size, field_size, field_size, -field_size],
        y=[-field_size, -field_size, field_size, field_size],
        z=[-0.001, -0.001, -0.001, -0.001],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color='green',
        opacity=0.5,
        name='Cricket Field'
    )

    # Return all pitch elements
    return [pitch_surface, pitch_outline, bowling_crease1, bowling_crease2] + popping_creases + [field_surface]


# Create player skeleton traces (for either batsman or bowler)
def create_skeleton_traces(frame_data, player_type="batsman"):
    traces = []

    # Set different colors for batsman and bowler
    if player_type == "batsman":
        line_color = 'red'
        marker_color = 'blue'
    else:  # bowler
        line_color = 'purple'
        marker_color = 'orange'

    # Add skeleton lines
    for i, j in SKELETON:
        traces.append(go.Scatter3d(
            x=[frame_data[i, 0], frame_data[j, 0]],
            y=[frame_data[i, 1], frame_data[j, 1]],
            z=[frame_data[i, 2], frame_data[j, 2]],
            mode='lines',
            line=dict(color=line_color, width=6),
            name=f'{player_type.capitalize()} Bone {i}-{j}',
            showlegend=False
        ))

    # Add joints as markers
    traces.append(go.Scatter3d(
        x=frame_data[:, 0],
        y=frame_data[:, 1],
        z=frame_data[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=marker_color,
        ),
        name=f'{player_type.capitalize()} Joints',
        showlegend=True
    ))

    return traces


# Create interactive 3D pose animation with both batsman and bowler
def create_interactive_dual_animation(batsman_frames, bowler_frames):
    # Create figure
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    # Determine the number of frames to use (minimum of both sequences)
    num_frames = min(len(batsman_frames), len(bowler_frames))

    # Get first frames to set up initial plot
    batsman_first_frame = batsman_frames[0]
    bowler_first_frame = bowler_frames[0]

    # Add cricket pitch elements
    pitch_elements = create_cricket_pitch()
    for element in pitch_elements:
        fig.add_trace(element)

    # Add initial batsman skeleton
    batsman_traces = create_skeleton_traces(batsman_first_frame, "batsman")
    for trace in batsman_traces:
        fig.add_trace(trace)

    # Add initial bowler skeleton
    # Position the bowler at the opposite end of the pitch
    adjusted_bowler_frame = bowler_first_frame.copy()
    adjusted_bowler_frame[:, 1] += 6.6  # Shift along y-axis to bowling end

    bowler_traces = create_skeleton_traces(adjusted_bowler_frame, "bowler")
    for trace in bowler_traces:
        fig.add_trace(trace)

    # Create frames for animation
    plotly_frames = []
    for frame_idx in range(num_frames):
        frame_traces = []

        # Add pitch elements (these don't change)
        for element in pitch_elements:
            frame_traces.append(element)

        # Get pose data for this frame
        batsman_data = batsman_frames[frame_idx]
        bowler_data = bowler_frames[frame_idx].copy()

        # Position the bowler at the opposite end of the pitch
        bowler_data[:, 1] += 6.5  # Shift along y-axis to bowling end

        # Add batsman skeleton traces
        batsman_frame_traces = create_skeleton_traces(batsman_data, "batsman")
        frame_traces.extend(batsman_frame_traces)

        # Add bowler skeleton traces
        bowler_frame_traces = create_skeleton_traces(bowler_data, "bowler")
        frame_traces.extend(bowler_frame_traces)

        # Add to frames list
        plotly_frames.append(go.Frame(
            data=frame_traces,
            name=str(frame_idx)
        ))

    # Add frames to figure
    fig.frames = plotly_frames

    # Configure layout
    fig.update_layout(
        title='Cricket 3D Animation - Batsman vs. Bowler',
        scene=dict(
            xaxis=dict(title='X Axis', range=[-4, 4]),
            yaxis=dict(title='Y Axis', range=[-1, 7]),
            zaxis=dict(title='Z Axis', range=[0, 6]),
            aspectmode='cube'
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 50, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.1,
            'y': 0,
            'xanchor': 'right'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 16},
                'prefix': 'Frame: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 50},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [
                        [str(i)],
                        {
                            'frame': {'duration': 50, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }
                    ],
                    'label': str(i),
                    'method': 'animate'
                }
                for i in range(num_frames)
            ]
        }]
    )

    return fig


# Function to synchronize batsman and bowler frames if they have different lengths
def synchronize_frames(batsman_frames, bowler_frames):
    # If they're the same length, return as is
    if len(batsman_frames) == len(bowler_frames):
        return batsman_frames, bowler_frames

    # Determine which one is shorter
    if len(batsman_frames) < len(bowler_frames):
        shorter_frames = batsman_frames
        longer_frames = bowler_frames
        is_batsman_shorter = True
    else:
        shorter_frames = bowler_frames
        longer_frames = batsman_frames
        is_batsman_shorter = False

    # Calculate the ratio to sample the longer sequence
    ratio = len(longer_frames) / len(shorter_frames)

    # Create a new synchronized sequence by sampling the longer one
    synchronized_longer = []
    for i in range(len(shorter_frames)):
        idx = min(int(i * ratio), len(longer_frames) - 1)
        synchronized_longer.append(longer_frames[idx])

    # Return in the original order
    if is_batsman_shorter:
        return batsman_frames, synchronized_longer
    else:
        return synchronized_longer, bowler_frames


# Main execution
def visualanimation():
    # Load batsman data
    batsman_json_path = "demo/3d_pose/sim/all_poses_3d.json"  # Update with your JSON file path
    batsman_data = load_pose_data(batsman_json_path)
    batsman_frames = extract_frames(batsman_data)

    # Load bowler data
    bowler_json_path = "demo/3d_pose/sim/all_poses_3d_bowl.json"  # Update with your JSON file path
    bowler_data = load_pose_data(bowler_json_path)
    bowler_frames = extract_frames(bowler_data)

    # Synchronize frames if they have different lengths
    batsman_frames, bowler_frames = synchronize_frames(batsman_frames, bowler_frames)

    # Create interactive animation with both players
    fig = create_interactive_dual_animation(batsman_frames, bowler_frames)

    # Show the figure in a browser or Jupyter notebook
    fig.show()

    # Save as HTML for sharing
    # pio.write_html(fig, "cricket_dual_animation.html")