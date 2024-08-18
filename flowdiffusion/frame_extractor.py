import os
import cv2

def extract_first_frame(video_path, frame_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if success:
        cv2.imwrite(frame_path, frame)
    cap.release()

def create_destination_folder_structure(src_path, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

def process_videos(src_folder, dst_folder):
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                src_video_path = os.path.join(root, file)
                
                # Determine the relative path and create destination path
                relative_path = os.path.relpath(root, src_folder)
                dst_folder_path = os.path.join(dst_folder, relative_path)
                create_destination_folder_structure(src_folder, dst_folder_path)
                
                # Determine the destination frame path
                dst_frame_path = os.path.join(dst_folder_path, os.path.splitext(file)[0] + '.jpg')
                
                # Extract and save the first frame
                extract_first_frame(src_video_path, dst_frame_path)
                print(f"Saved first frame of {src_video_path} to {dst_frame_path}")

# Set your source and destination folders
source_folder = '/home/cobot/testing/AVDC/datasets/vidgen_datasets'
destination_folder = '/home/cobot/testing/AVDC/results/bridge/first_frames'

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

# Process the videos
process_videos(source_folder, destination_folder)
