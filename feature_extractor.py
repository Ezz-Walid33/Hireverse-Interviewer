import sys
import os
import tempfile
import subprocess
import shutil
import wave

# Set the base directory to Hireverse-Interviewer
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

from hireverse.utils.face_analyzer import FaceAnalyzer
from hireverse.schemas.frame import Frame
from hireverse.utils.feature_storage import FeatureStorage
from hireverse.schemas.model_features import ProsodicFeatures
from hireverse.utils.prosody_analyzer import ProsodyAnalyzer
from hireverse.utils.LexicalAnalyser import LexicalAnalyser
import numpy as np

def extract_audio_from_video(video_path, output_wav_path):
    """Extracts audio from a video file using ffmpeg."""
    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_wav_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("ffmpeg error:", result.stderr.decode())

def is_valid_wav(filepath):
    try:
        with wave.open(filepath, 'rb') as wav_file:
            return wav_file.getnframes() > 0
    except Exception:
        return False

# Ensure output directory exists
output_dir = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(output_dir, exist_ok=True)

def extract_features_from_video(participant_id, video_path, output_csv_path=None):
    print(f"[INFO] Starting feature extraction for participant: {participant_id}")
    print(f"[INFO] Video path: {video_path}")

    if output_csv_path is None:
        output_csv_path = os.path.join(BASE_DIR, "data", "processed", "interview_features.csv")
    print(f"[INFO] Output CSV path: {output_csv_path}")

    face_analyzer = FaceAnalyzer()
    print("[INFO] Extracting video frames...")
    frames = face_analyzer.get_video_frames(
        video_path, participant_id
    )
    print(f"[INFO] Number of frames extracted: {len(frames)}")

    # Process facial landmarks
    filtered_frames = []
    print("[INFO] Processing facial landmarks...")
    for frame in frames:
        frame.facial_landmarks_obj = face_analyzer.process_image_results(frame.image)
        if frame.facial_landmarks_obj:
            frame.facial_landmarks = frame.facial_landmarks_obj.landmark
            filtered_frames.append(frame)
    frames = filtered_frames
    print(f"[INFO] Frames with valid facial landmarks: {len(frames)}")

    # Face coordinates
    print("[INFO] Extracting face coordinates...")
    for frame in frames:
        if frame.facial_landmarks:
            frame.face = face_analyzer.get_face_coordinates(frame.facial_landmarks, frame.image)

    # Head displacement
    print("[INFO] Calculating head displacement...")
    for i in range(len(frames) - 1):
        bbox1 = face_analyzer.get_bouding_box_center(frames[i].face)
        bbox2 = face_analyzer.get_bouding_box_center(frames[i + 1].face)
        frames[i].head_displacement = face_analyzer.get_displacement_between_two_bounding_boxes(bbox1, bbox2)
        frames[i].head_vertical_displacement = face_analyzer.get_vertical_displacement_between_two_bounding_boxes(bbox1, bbox2)
        frames[i].head_horizontal_displacement = face_analyzer.get_horizontal_distance_between_two_bounding_boxes(bbox1, bbox2)

    # Smile (happiness)
    print("[INFO] Calculating smile probabilities...")
    SMOOTH_WINDOW = 5
    happiness_buffer = []
    def smooth_happiness(happiness_prob):
        if happiness_prob is None:
            return 0
        happiness_buffer.append(happiness_prob)
        if len(happiness_buffer) > SMOOTH_WINDOW:
            happiness_buffer.pop(0)
        return np.mean(happiness_buffer)

    for i, frame in enumerate(frames):
        face_roi = face_analyzer.get_face_roi_image(frame.image, frame.face, expand_ratio=1.1)
        frame.smile = smooth_happiness(face_analyzer.get_smile_from_frame(face_roi))

    # Selected facial features
    print("[INFO] Extracting selected facial features...")
    for frame in frames:
        frame.two_landmarks_connectors = face_analyzer.get_selected_facial_landmarks(frame.facial_landmarks)

    # Head pose
    print("[INFO] Calculating head pose angles...")
    for frame in frames:
        result = face_analyzer.get_face_angles(frame.image, frame.facial_landmarks)
        frame.face_angles = result

    # --- Extract audio from video and analyze prosody & lexical ---
    audio_dir = os.path.join(BASE_DIR, "data", "raw", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    final_audio_path = os.path.join(audio_dir, f"trimmed_{participant_id}.wav")

    print(f"[INFO] Extracting audio from video to: {final_audio_path}")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=audio_dir) as temp_audio:
        temp_audio_path = temp_audio.name
    try:
        extract_audio_from_video(video_path, temp_audio_path)
        print(f"[INFO] Audio extracted to temp file: {temp_audio_path}")
        shutil.move(temp_audio_path, final_audio_path)
        print(f"[INFO] Audio moved to final path: {final_audio_path}")

        # --- COPY FOR ProsodyAnalyzer BEFORE running it ---
        venv_src_audio_dir = os.path.abspath(
            os.path.join(BASE_DIR, ".venv", "src", "hireverse", "data", "raw", "audio")
        )
        os.makedirs(venv_src_audio_dir, exist_ok=True)
        venv_expected_audio_path = os.path.join(
            venv_src_audio_dir, f"trimmed_{participant_id}.wav"
        )
        shutil.copy(final_audio_path, venv_expected_audio_path)
        print(f"[INFO] Audio copied to venv src path for ProsodyAnalyzer: {venv_expected_audio_path}")
        print(f"[DEBUG] File exists before ProsodyAnalyzer? {os.path.exists(venv_expected_audio_path)}")

        # now run prosody & lexical
        print("[INFO] Running prosody analysis...")
        try:
            prosody_analyzer = ProsodyAnalyzer(participant_id)
            prosodic_features = prosody_analyzer.extract_all_features()
            print("[INFO] Prosody analysis complete.")
        except Exception as e:
            print(f"[ERROR] Prosody analysis failed: {e}")
            import traceback
            traceback.print_exc()
            prosodic_features = None

        print("[INFO] Running lexical analysis...")
        lexical_analyser = None
        try:
            lexical_analyser = LexicalAnalyser(final_audio_path)
            lexical_features = lexical_analyser.extract_all_features()
            print("[INFO] Lexical analysis complete.")
        except Exception as e:
            print(f"[ERROR] Lexical analysis failed: {e}")
            import traceback
            traceback.print_exc()
            lexical_features = None

        transcript_txt = os.path.join(output_dir, f"{participant_id}_transcript.txt")
        if lexical_analyser is not None and hasattr(lexical_analyser, "transcript") and lexical_analyser.transcript:
            with open(transcript_txt, "w", encoding="utf-8") as f:
                f.write(lexical_analyser.transcript)
            print(f"[INFO] Transcript saved to {transcript_txt}")
    finally:
        # only clean up after ALL processing is done
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
    # Aggregate and save
    print("[INFO] Aggregating and saving features...")



    print(f"[DEBUG] Number of frames passed to aggregation: {len(frames)}")
    if len(frames) == 0:
        print("[ERROR] No frames available for aggregation. Skipping aggregation and save.")
        facial_features = None
        # Save empty CSVs for consistency
        facial_csv = os.path.join(output_dir, f"{participant_id}_facial_features.csv")
        prosodic_csv = os.path.join(output_dir, f"{participant_id}_prosodic_features.csv")
        lexical_csv = os.path.join(output_dir, f"{participant_id}_lexical_features.csv")
        import pandas as pd
        pd.DataFrame().to_csv(facial_csv, index=False)
        pd.DataFrame().to_csv(prosodic_csv, index=False)
        pd.DataFrame().to_csv(lexical_csv, index=False)
        return {
            "facial_features": facial_features,
            "prosodic_features": prosodic_features,
            "lexical_features": lexical_features,
            "error": "No frames available for aggregation."
        }
    else:
        feature_storage = FeatureStorage(output_csv_path)
        facial_features = feature_storage.aggregate_facial_features(frames)
        # Save facial features
        facial_csv = os.path.join(output_dir, f"{participant_id}_facial_features.csv")
        import pandas as pd
        pd.DataFrame([facial_features]).to_csv(facial_csv, index=False)
        # Save prosodic features
        prosodic_csv = os.path.join(output_dir, f"{participant_id}_prosodic_features.csv")
        if prosodic_features is not None:
            pd.DataFrame([prosodic_features]).to_csv(prosodic_csv, index=False)
        else:
            pd.DataFrame().to_csv(prosodic_csv, index=False)
        # Save lexical features
        lexical_csv = os.path.join(output_dir, f"{participant_id}_lexical_features.csv")
        if lexical_features is not None:
            pd.DataFrame([lexical_features]).to_csv(lexical_csv, index=False)
        else:
            pd.DataFrame().to_csv(lexical_csv, index=False)

        print("[INFO] Feature extraction complete.")

        # now that everything is written, remove audio copies
        for p in (final_audio_path, venv_expected_audio_path):
            if os.path.exists(p):
                os.remove(p)

        return {
            "facial_features": facial_features,
            "prosodic_features": prosodic_features,
            "lexical_features": lexical_features
        }
