import re
import time

import torch
from utils.audio_generator import AudioGenerator
from utils.audio_utils import combine_wav_tracks
from utils.directory_utils import prepare_directory
from utils.text_utils import extract_text_from_docx, split_sentence

# Initial data
file_name = "resources/texts/text-1.txt"
sentences_in_one_file = 1000

references_directory = "resources/voice-cloning/mihail"
directory_parent_name = "resources/generated"
directory_name = "ru-warm-mihail"
# force_emotion = "neutral"
language = "ru"

is_word_file = file_name.endswith(".docx")

if is_word_file:
    text_from_file = extract_text_from_docx(file_name)
else:
    text_from_file = open(file_name, "r").read()

sentences = [sentence.rstrip('.?!\n') for sentence in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text_from_file)]
sentences_count = len(sentences)

sentence_files_count = sentences_count // sentences_in_one_file + 1
print(f"Target files count: {sentence_files_count} (for {sentences_count} sentences)")

directory_path = f"{directory_parent_name}/{directory_name}"
prepare_directory(directory_path)

device = "cuda" if torch.cuda.is_available() else "cpu"

start_time = time.time()

audio_generator = AudioGenerator(directory_path, references_directory, device)

for file_index in range(sentence_files_count):
    file_directory = f"{directory_path}/{file_index}"
    prepare_directory(file_directory)

    for local_sentence_index in range(sentences_in_one_file):
        sentence_index = file_index * sentences_in_one_file + local_sentence_index
        if sentence_index >= sentences_count:
            break

        sentence = sentences[sentence_index]
        is_sentence_empty = len(sentence.strip()) == 0
        if is_sentence_empty:
            continue

        print(f"Generating audio for sentence {sentence_index + 1}/{len(sentences)}")

        if len(sentence) <= 182:
            audio_generator.generate_audio_file(sentence, f"{file_index}/{local_sentence_index}", language)
        else:
            sentence_parts = split_sentence(sentence)
            for j, sentence_part in enumerate(sentence_parts):
                if sentence_part:
                    print(f"Generating audio for sentence part {j + 1}/{len(sentence_parts)}")
                    audio_generator.generate_audio_file(sentence_part, f"{file_index}/{local_sentence_index}.{j}", language)

        print("")

    combine_wav_tracks(f"{directory_path}/{file_index}")


end_time = time.time()
print(f"Total execution time: {end_time - start_time} seconds")