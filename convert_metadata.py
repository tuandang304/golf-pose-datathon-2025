import os
import re
import cv2
import pandas as pd

def snake_case(s):
    return '_'.join(
        re.sub('([A-Z][a-z]+)', r' \1',
        re.sub('([A-Z]+)', r' \1',
        s.replace('-', ' '))).split()).lower()

def build_metadata(dataset_path="CUSTOM_DATASET", video_metadata_path="video_metadata.csv", error_metadata_path="error_metadata.csv"):
    video_metadata = []
    
    for view in os.listdir(dataset_path):
        view_path = os.path.join(dataset_path, view)
        if not os.path.isdir(view_path):
            continue
            
        for swing_quality in os.listdir(view_path):
            swing_quality_path = os.path.join(view_path, swing_quality)
            if not os.path.isdir(swing_quality_path):
                continue

            direct_swing_path = os.path.join(swing_quality_path)
            for video_name in os.listdir(direct_swing_path):
                video_path = os.path.join(direct_swing_path, video_name)
                if os.path.isdir(video_path):
                    error_name = snake_case(os.path.basename(video_path))
                    for sub_video_name in os.listdir(video_path):
                        sub_video_path = os.path.join(video_path, sub_video_name)
                        if sub_video_path.endswith((".mp4", ".avi", ".mov")):
                             video_metadata.append(process_video(sub_video_path, view, swing_quality, error_name))
                elif video_path.endswith((".mp4", ".avi", ".mov")):
                    error_name = "none"
                    video_metadata.append(process_video(video_path, view, swing_quality, error_name))


    video_df = pd.DataFrame(video_metadata)
    video_df.to_csv(video_metadata_path, index=False)
    print(f"Video metadata saved to {video_metadata_path}")

    # Create error metadata
    error_df = video_df[video_df["error_name"] != "none"].copy()
    error_summary = error_df.groupby("error_name")["duration_sec"].agg(['count', 'mean', 'min', 'max']).reset_index()
    error_summary.rename(columns={'count': 'video_count'}, inplace=True)
    error_summary["type"] = error_summary["mean"].apply(lambda x: "static" if x < 3 else "dynamic")
    error_summary.to_csv(error_metadata_path, index=False)
    print(f"Error metadata saved to {error_metadata_path}")

def process_video(video_path, view, swing_quality, error_name):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0
    cap.release()

    return {
        "view": view,
        "swing_quality": swing_quality,
        "error_name": error_name,
        "video_path": video_path,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }

if __name__ == "__main__":
    build_metadata()