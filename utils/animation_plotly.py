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
        y=[0,0],
        z=[0.001, 0.001],
        mode='lines',
        line=dict(color='white', width=4),
        name='Bowling Crease 1'
    )

    # Bowling crease at other end
    bowling_crease2 = go.Scatter3d(
        x=[-pitch_width / 2 - 0.05, pitch_width / 2 + 0.05],
        y=[pitch_length, pitch_length],
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


# Create interactive 3D pose animation with Plotly
def create_interactive_pose_animation(frames):
    # Create figure
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    # Get first frame to set up initial plot
    first_frame = frames[0]

    # Add cricket pitch elements
    pitch_elements = create_cricket_pitch()
    for element in pitch_elements:
        fig.add_trace(element)

    # Add skeleton lines and joints for the first frame
    for i, j in SKELETON:
        fig.add_trace(go.Scatter3d(
            x=[first_frame[i, 0], first_frame[j, 0]],
            y=[first_frame[i, 1], first_frame[j, 1]],
            z=[first_frame[i, 2], first_frame[j, 2]],
            mode='lines',
            line=dict(color='red', width=6),
            name=f'Bone {i}-{j}',
            showlegend=False
        ))

    # Add joints as markers
    fig.add_trace(go.Scatter3d(
        x=first_frame[:, 0],
        y=first_frame[:, 1],
        z=first_frame[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color='blue',
        ),
        name='Joints',
        showlegend=True
    ))

    # Create frames for animation
    plotly_frames = []
    for frame_idx, frame_data in enumerate(frames):
        frame_traces = []

        # Add pitch elements (these don't change)
        for element in pitch_elements:
            frame_traces.append(element)

        # Add skeleton lines for this frame
        for i, j in SKELETON:
            frame_traces.append(go.Scatter3d(
                x=[frame_data[i, 0], frame_data[j, 0]],
                y=[frame_data[i, 1], frame_data[j, 1]],
                z=[frame_data[i, 2], frame_data[j, 2]],
                mode='lines',
                line=dict(color='red', width=6),
                showlegend=False
            ))

        # Add joints for this frame
        frame_traces.append(go.Scatter3d(
            x=frame_data[:, 0],
            y=frame_data[:, 1],
            z=frame_data[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color='blue',
            ),
            showlegend=False
        ))

        # Add to frames list
        plotly_frames.append(go.Frame(
            data=frame_traces,
            name=str(frame_idx)
        ))

    # Add frames to figure
    fig.frames = plotly_frames

    # Configure layout
    fig.update_layout(
        title='Cricket Batsman 3D Pose Animation',
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
                for i in range(len(frames))
            ]
        }]
    )

    return fig


# Main execution
def main():
    json_path = "demo/3d_pose/sim/all_poses_3d.json"  # Update with your JSON file path
    data = load_pose_data(json_path)
    frames = extract_frames(data)

    # Create interactive animation
    fig = create_interactive_pose_animation(frames)

    # Show the figure in a browser or Jupyter notebook
    fig.show()

    # Save as HTML for sharing
    # pio.write_html(fig, "cricket_pose_animation.html")

    # Save as video (requires additional dependencies)
    # pio.write_image(fig, "cricket_pose_animation.mp4")


if __name__ == "__main__":
    main()