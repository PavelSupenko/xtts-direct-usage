import os
import wave


def numerical_sort(value):
    parts = value.split('.')
    numeric_parts = []
    for part in parts:
        try:
            numeric_parts.append(int(part))
        except ValueError:
            numeric_parts.append(part)
    return numeric_parts


def combine_wav_tracks(directory):
    directory_files = os.listdir(directory)
    directory_files.sort(key=numerical_sort)
    wav_files = [f"{directory}/{f}" for f in directory_files if f.endswith(".wav")]
    result_file = f"{directory}/combined.wav"

    if len(wav_files) == 0:
        return

    data = []
    for infile in wav_files:
        w = wave.open(infile, 'rb')
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()

    output = wave.open(result_file, 'wb')
    output.setparams(data[0][0])
    for i in range(len(data)):
        output.writeframes(data[i][1])
    output.close()
