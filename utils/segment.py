from scenedetect import SceneManager, open_video, detect, ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg


def segment_video(video_path, output_dir):
    # Input and output file paths
    # video_path = "Cric Test2.mp4"
    # output_dir = "segments_2"

    # Open video and set up scene manager
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    # Detect scenes
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    # Print detected scenes
    print(f"Detected {len(scene_list)} scenes.")
    for i, (start, end) in enumerate(scene_list):
        print(f"Scene {i + 1}: Start {start.get_timecode()} - End {end.get_timecode()}")

    # Split video based on detected scenes
    split_video_ffmpeg(video_path, scene_list, output_dir)
    print("Segmentation complete. Clips saved in:", output_dir)

