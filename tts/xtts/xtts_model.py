import os.path
import time
from typing import List, Dict, Any

import pickle
import librosa
import numpy as np
import torch
import torchaudio

import torch.nn.functional as functional

from tts.audio.numpy_transforms import save_wav
from tts.gpt.gpt import GPT
from tts.utils.io import load_fsspec
from tts.xtts.hifigan_decoder import HifiDecoder
from tts.xtts.voice_bpe_tokenizer import VoiceBpeTokenizer

from trainer import TrainerModel

from tts.configs.xttc_config_unpickler import XttsConfigUnpickler

# init_stream_support()

class XttsModel(TrainerModel):
    def __init__(self, model_path, device):
        super().__init__()

        self.model_path = model_path
        self.device = device

        self.speakers_model_path = os.path.join(model_path, "speakers_xtts.pth")
        self.synthesizer_model_path = os.path.join(model_path, "model.pth")
        self.vocabulary_json_path = os.path.join(model_path, "vocab.json")

        self.register_buffer("mel_stats", torch.ones(80))
        self.tokenizer = VoiceBpeTokenizer(vocab_file=self.vocabulary_json_path)

        # All this hardcoded values are from the config.json file
        self.gpt_use_perceiver_resampler = True

        self. output_sample_rate = 24000
        self. sample_rate = 22050

        self.hifigan_decoder = HifiDecoder(
            input_sample_rate=22050,  # self.args.input_sample_rate
            output_sample_rate=24000,  # self.args.output_sample_rate
            output_hop_length=256,  # self.args.output_hop_length
            ar_mel_length_compression=1024,  # self.args.gpt_code_stride_len
            decoder_input_dim=1024, # self.args.decoder_input_dim
            d_vector_dim=512,  # self.args.d_vector_dim
            cond_d_vector_in_each_upsampling_layer=True,  # self.args.cond_d_vector_in_each_upsampling_layer
        )

        self.gpt_number_text_tokens = 6681
        self.gpt_max_text_tokens = 402
        self.gpt_batch_size = 1

        self.gpt_number_text_tokens = self.tokenizer.get_number_tokens()
        self.gpt_start_text_token = self.tokenizer.tokenizer.token_to_id("[START]")
        self.gpt_stop_text_token = self.tokenizer.tokenizer.token_to_id("[STOP]")

        self.gpt = GPT(
            layers=30,  # self.args.gpt_layers
            model_dim=1024,  # self.args.gpt_n_model_channels
            start_text_token=self.gpt_start_text_token,
            stop_text_token=self.gpt_stop_text_token,
            heads=16,  # self.args.gpt_n_heads,
            max_text_tokens=402,  # self.args.gpt_max_text_tokens,
            max_mel_tokens=605,  # self.args.gpt_max_audio_tokens,
            max_prompt_tokens=70, # self.args.gpt_max_prompt_tokens,
            number_text_tokens=self.gpt_number_text_tokens,
            num_audio_tokens=1026,  # self.args.gpt_num_audio_tokens,
            start_audio_token=1024,  # self.args.gpt_start_audio_token,
            stop_audio_token=1025,  # self.args.gpt_stop_audio_token,
            use_perceiver_resampler=True,  # self.args.gpt_use_perceiver_resampler,
            code_stride_len=1024,  # self.args.gpt_code_stride_len,
        )

        kv_cache = True  # self.args.kv_cache
        strict = True
        use_deepspeed = False

        checkpoint = self.get_compatible_checkpoint_state_dict(self.synthesizer_model_path)

        # First we need to load dictionary data from pytorch model and only after that
        # We can eval() modules because if we are doing that vise versa the pytourch
        # serializer will fail looking for the dictionary data
        self.load_state_dict(checkpoint, strict=strict)

        self.hifigan_decoder.eval()
        self.gpt.init_gpt_for_inference(kv_cache=kv_cache, use_deepspeed=use_deepspeed)
        self.gpt.eval()


    def get_compatible_checkpoint_state_dict(self, model_path):
        pickle.Unpickler = XttsConfigUnpickler
        checkpoint = load_fsspec(model_path, map_location=torch.device("cpu"), pickle_module=pickle)["model"]
        # remove xtts gpt trainer extra keys
        ignore_keys = ["torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae", "dvae"]
        for key in list(checkpoint.keys()):
            # check if it is from the coqui Trainer if so convert it
            if key.startswith("xtts."):
                new_key = key.replace("xtts.", "")
                checkpoint[new_key] = checkpoint[key]
                del checkpoint[key]
                key = new_key

            # remove unused keys
            if key.split(".")[0] in ignore_keys:
                del checkpoint[key]

        return checkpoint

    def save_wav(self, wav: List[int], path: str, pipe_out=None) -> None:
        """Save the waveform as a file.

        Args:
            wav (List[int]): waveform as a list of values.
            path (str): output path to save the waveform.
            pipe_out (BytesIO, optional): Flag to stdout the generated TTS wav file for shell pipe.
        """
        # if tensor convert to numpy
        if torch.is_tensor(wav):
            wav = wav.cpu().numpy()
        if isinstance(wav, list):
            wav = np.array(wav)
        save_wav(wav=wav, path=path, sample_rate=self.output_sample_rate, pipe_out=pipe_out)

    def tts_to_file(
            self,
            text: str,
            language: str,
            speaker_wav: str,
            file_path: str
    ):
        wav = self.tts(
            text=text,
            language=language,
            speaker_wav=speaker_wav
        )
        self.save_wav(wav=wav, path=file_path)

    def tts(
            self,
            text: str,
            language: str,
            speaker_wav: str
    ):
        start_time = time.time()

        # from synthesizer
        wavs = []
        vocoder_device = self.device  # cpu or cuda

        # then we are going to 'if hasattr(self.tts_model, "synthesize")'
        # that uses self.tts_model.synthesize (from XTTS model class)

        # debug value
        # {'d_vector': None, 'gpt_cond_chunk_len': 4, 'gpt_cond_len': 30, 'length_penalty': 1.0, 'max_ref_len': 30, 'repetition_penalty': 5.0, 'sound_norm_refs': False, 'temperature': 0.75, 'top_k': 50, 'top_p': 0.85, 'voice_dirs': None}
        outputs = self.full_inference(text, speaker_wav, language)
        waveform = outputs["wav"]
        waveform = waveform.squeeze()

        # todo: add it as a parameter or remove totally
        # trim_silence = False
        # if trim_silence:
        #     waveform = self.trim_silence(waveform, self.tts_model.ap)

        wavs += list(waveform)
        wavs += [0] * 10000

        process_time = time.time() - start_time
        audio_time = len(wavs) / self.sample_rate
        print(f"Processing time: {process_time:.3f}")
        print(f"Real-time factor: {process_time / audio_time:.3f}")
        return wavs

    @torch.inference_mode()
    def full_inference(
            self,
            text,
            ref_audio_path,
            language,
            # GPT inference
            temperature=0.75,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=50,
            top_p=0.85,
            do_sample=True,
            # Cloning
            gpt_cond_len=30,
            gpt_cond_chunk_len=6,
            max_ref_len=10,
            sound_norm_refs=False,
            **hf_generate_kwargs,
    ):
        (gpt_cond_latent, speaker_embedding) = self.get_conditioning_latents(
            audio_path=ref_audio_path,
            gpt_cond_len=gpt_cond_len,
            gpt_cond_chunk_len=gpt_cond_chunk_len,
            max_ref_length=max_ref_len,
            sound_norm_refs=sound_norm_refs,
        )

        return self.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            **hf_generate_kwargs,
        )

    @torch.inference_mode()
    def inference(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        num_beams=1,
        speed=1.0,
        enable_text_splitting=False,
        **hf_generate_kwargs,
    ):
        language = language.split("-")[0]  # remove the country code
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)
        text = [text]

        wavs = []
        gpt_latents_list = []
        for sent in text:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to(self.device)

            assert (
                text_tokens.shape[-1] < self.gpt_max_text_tokens
            ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

            # The main work is done here!
            with torch.no_grad():
                gpt_codes = self.gpt.generate(
                    cond_latents=gpt_cond_latent,
                    text_inputs=text_tokens,
                    input_tokens=None,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=self.gpt_batch_size,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    output_attentions=False,
                    **hf_generate_kwargs,
                )
                expected_output_len = torch.tensor(
                    [gpt_codes.shape[-1] * self.gpt.code_stride_len], device=text_tokens.device
                )

                text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
                gpt_latents = self.gpt(
                    text_tokens,
                    text_len,
                    gpt_codes,
                    expected_output_len,
                    cond_latents=gpt_cond_latent,
                    return_attentions=False,
                    return_latent=True,
                )

                if length_scale != 1.0:
                    gpt_latents = functional.interpolate(
                        gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear"
                    ).transpose(1, 2)

                gpt_latents_list.append(gpt_latents.cpu())
                wavs.append(self.hifigan_decoder(gpt_latents, g=speaker_embedding).cpu().squeeze())

        return {
            "wav": torch.cat(wavs, dim=0).numpy(),
            "gpt_latents": torch.cat(gpt_latents_list, dim=1).numpy(),
            "speaker_embedding": speaker_embedding,
        }

    @torch.inference_mode()
    def get_conditioning_latents(
            self,
            audio_path,
            max_ref_length=30,
            gpt_cond_len=6,
            gpt_cond_chunk_len=6,
            librosa_trim_db=None,
            sound_norm_refs=False,
            load_sr=22050,
    ):
        # todo: all time has 1 item. Needs to be removed
        speaker_embeddings = []

        audio = self.load_audio(audio_path, load_sr)
        audio = audio[:, : load_sr * max_ref_length].to(self.device)
        if sound_norm_refs:
            audio = (audio / torch.abs(audio).max()) * 0.75
        if librosa_trim_db is not None:
            audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

        # compute latents for the decoder
        speaker_embedding = self.get_speaker_embedding(audio, load_sr)
        speaker_embeddings.append(speaker_embedding)

        gpt_cond_latents = self.get_gpt_cond_latents(
            audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
        )  # [1, 1024, T]

        if speaker_embeddings:
            speaker_embedding = torch.stack(speaker_embeddings)
            speaker_embedding = speaker_embedding.mean(dim=0)

        return gpt_cond_latents, speaker_embedding

    @torch.inference_mode()
    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 6):
        """Compute the conditioning latents for the GPT model from the given audio.

        Args:
            audio (tensor): audio tensor.
            sr (int): Sample rate of the audio.
            length (int): Length of the audio in seconds. If < 0, use the whole audio. Defaults to 30.
            chunk_length (int): Length of the audio chunks in seconds. When `length == chunk_length`, the whole audio
                is being used without chunking. It must be < `length`. Defaults to 6.
        """
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, : 22050 * length]
        if self.gpt_use_perceiver_resampler:
            style_embs = []
            for i in range(0, audio.shape[1], 22050 * chunk_length):
                audio_chunk = audio[:, i: i + 22050 * chunk_length]

                # if the chunk is too short ignore it
                if audio_chunk.size(-1) < 22050 * 0.33:
                    continue

                mel_chunk = self.wav_to_mel_cloning(
                    audio_chunk,
                    mel_norms=self.mel_stats.cpu(),
                    n_fft=2048,
                    hop_length=256,
                    win_length=1024,
                    power=2,
                    normalized=False,
                    sample_rate=22050,
                    f_min=0,
                    f_max=8000,
                    n_mels=80,
                )
                style_emb = self.gpt.get_style_emb(mel_chunk.to(self.device), None)
                style_embs.append(style_emb)

            # mean style embedding
            cond_latent = torch.stack(style_embs).mean(dim=0)
        else:
            mel = self.wav_to_mel_cloning(
                audio,
                mel_norms=self.mel_stats.cpu(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                power=2,
                normalized=False,
                sample_rate=22050,
                f_min=0,
                f_max=8000,
                n_mels=80,
            )
            cond_latent = self.gpt.get_style_emb(mel.to(self.device))
        return cond_latent.transpose(1, 2)

    @torch.inference_mode()
    def get_speaker_embedding(self, audio, sr):
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return (
            self.hifigan_decoder.speaker_encoder.forward(audio_16k.to(self.device), l2_norm=True)
            .unsqueeze(-1)
            .to(self.device)
        )

    @staticmethod
    def wav_to_mel_cloning(
            wav,
            mel_norms_file="../experiments/clips_mel_norms.pth",
            mel_norms=None,
            device=torch.device("cpu"),
            n_fft=4096,
            hop_length=1024,
            win_length=4096,
            power=2,
            normalized=False,
            sample_rate=22050,
            f_min=0,
            f_max=8000,
            n_mels=80,
    ):
        """
        Convert waveform to mel-spectrogram with hard-coded parameters for cloning.

        Args:
            wav (torch.Tensor): Input waveform tensor.
            mel_norms_file (str): Path to mel-spectrogram normalization file.
            mel_norms (torch.Tensor): Mel-spectrogram normalization tensor.
            device (torch.device): Device to use for computation.

        Returns:
            torch.Tensor: Mel-spectrogram tensor.
        """
        mel_stft = torchaudio.transforms.MelSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=power,
            normalized=normalized,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            norm="slaney",
        ).to(device)
        wav = wav.to(device)
        mel = mel_stft(wav)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if mel_norms is None:
            mel_norms = torch.load(mel_norms_file, map_location=device)
        mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
        return mel

    @staticmethod
    def load_audio(audiopath, sampling_rate):
        # better load setting following: https://github.com/faroit/python_audio_loading_benchmark

        # torchaudio should chose proper backend to load audio depending on platform
        audio, lsr = torchaudio.load(audiopath)

        # stereo to mono if needed
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        if lsr != sampling_rate:
            audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

        # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
        # '10' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
        if torch.any(audio > 10) or not torch.any(audio < 0):
            print(f"Error with {audiopath}. Max={audio.max():.2f} min={audio.min():.2f}")
        # clip audio invalid values
        audio.clip_(-1, 1)
        return audio

    def get_data_loader(*args: Any, **kwargs: Any) -> torch.utils.data.DataLoader:
        pass

    def forward(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://coqui-tts.readthedocs.io/en/dev/models/xtts.html#training"
        )