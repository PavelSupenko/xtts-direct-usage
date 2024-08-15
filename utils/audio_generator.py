import os
from tts.xtts.xtts_model import XttsModel
from utils.text_utils import detect_emotion


class AudioGenerator:
    def __init__(self, base_output_directory: str, emotions_directory: str, device: str = "cpu"):
        self.base_output_directory = base_output_directory
        self.emotions_directory = emotions_directory

        self.xtts_model = XttsModel("resources/xtts2", device)

    def generate_audio_file(self, text, file_name, language, force_emotion=None):
        if force_emotion:
            emotion_file = self.get_emotion_audio_file(force_emotion, 0.0)
            print(f"Force using emotion: {force_emotion} file: {emotion_file}) will be used")
        else:
            emotion, score = detect_emotion(text, language, min_score=0.6)
            emotion_file = self.get_emotion_audio_file(emotion, score)
            print(f"Detected emotion: {emotion} (score: {score}) file: {emotion_file}) will be used")

        file_path = f"{self.base_output_directory}/{file_name}.wav"

        self.xtts_model.tts_to_file(
            text=text,
            language=language,
            speaker_wav=emotion_file,
            file_path=file_path)

    def get_emotion_audio_file(self, emotion: str, score: float,
            first_grade_max_score: float = 0.6,
            second_grade_max_score: float = 0.8) -> str:

        emotion_directory = f"{self.emotions_directory}/{emotion}"
        # todo: Add support for not only directories but single named audio files

        emotion_files = os.listdir(emotion_directory)
        emotion_wav_files = [f for f in emotion_files if f.endswith(".wav")]
        emotion_wav_files.sort()
        wav_files_count = len(emotion_wav_files)

        if wav_files_count == 1:
            emotion_file_index = 0
        else:
            if score < first_grade_max_score:
                emotion_file_index = 0
            elif wav_files_count == 2 or score < second_grade_max_score:
                emotion_file_index = 1
            else:
                emotion_file_index = 2

        emotion_file = f"{emotion_directory}/{emotion_wav_files[emotion_file_index]}"
        return emotion_file
