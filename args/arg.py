import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Cricket Pipeline Components")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input video file"
    )

    parser.add_argument(
        "--rally_segment",
        action="store_true",
        help="Enable rally segmentation module"
    )

    parser.add_argument(
        "--player_det",
        action="store_true",
        help="Enable player detection module"
    )

    parser.add_argument(
        "--pose_est",
        action="store_true",
        help="Enable pose estimation module"
    )

    parser.add_argument(
        "--animate",
        action="store_true",
        help="Enable animation module"
    )

    return parser.parse_args()
