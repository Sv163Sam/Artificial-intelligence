import os
import cv2

# PROJECT_PATHS
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
RESULT_POSES_PATH = os.path.join(ROOT_PATH, "Analyze_results/Poses")
RESULT_CENTERS_PATH = os.path.join(ROOT_PATH, "Analyze_results/Centers")
RESULT_TRANSLATED_CENTERS_PATH = os.path.join(ROOT_PATH, "Analyze_results/Translated_centers")
SOURCE_PATH = os.path.join(ROOT_PATH, "Analyze_source/Source")


# UTILS
def count_video_files(directory: str):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    count = 0

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            if any(filename.endswith(ext) for ext in video_extensions):
                count += 1

    return count


def count_frames(video_base_size: int, source_path: str) -> list | None:
    count_frames_in_files = []

    for file_index in range(0, video_base_size):
        path = os.path.join(source_path, f"{file_index + 1}.txt")

        with open(path, 'r') as file:
            line_count = sum(1 for line in file)
            count_frames_in_files.append(line_count)

    return count_frames_in_files
