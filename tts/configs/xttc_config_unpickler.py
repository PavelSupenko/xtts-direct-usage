import pickle


class XttsConfigUnpickler(pickle.Unpickler):
    """
    Overload default pickler to solve XTTS config naming problem
    It's looking for mod_name='TTS.tts.configs.xtts_config' and name='XttsConfig'
    """

    def find_class(self, module, name):
        if module == 'TTS.tts.configs.xtts_config' and name == 'XttsConfig':
            return super().find_class("tts.configs.xtts_config", name)

        if module == 'TTS.tts.models.xtts' and name == 'XttsAudioConfig':
            return super().find_class("tts.configs.xtts_audio_config", name)

        if module == 'TTS.config.shared_configs' and name == 'BaseDatasetConfig':
            return super().find_class("tts.configs.base_dataset_config", name)

        if module == 'TTS.tts.models.xtts' and name == 'XttsArgs':
            return super().find_class("tts.configs.xtts_args", name)

        return super().find_class(module, name)
