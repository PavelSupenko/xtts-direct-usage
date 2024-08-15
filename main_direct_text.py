import torch

from utils.audio_generator import AudioGenerator
from utils.directory_utils import prepare_directory

device = "cuda" if torch.cuda.is_available() else "cpu"
references_directory = "resources/voice-cloning/pavlo"
directory_parent_name = "resources/generated"
directory_name = "mixed-pavlo"
language = "en"

sentence = "Hello, how are you?"

directory_path = f"{directory_parent_name}/{directory_name}"
prepare_directory(directory_path)

audio_generator = AudioGenerator(directory_path, references_directory, device)
audio_generator.generate_audio_file(sentence, f"xtts", language)
