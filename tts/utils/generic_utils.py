# -*- coding: utf-8 -*-
import datetime
import importlib
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# TODO: This method is duplicated in Trainer but out of date there
def get_git_branch():
    try:
        out = subprocess.check_output(["git", "branch"]).decode("utf8")
        current = next(line for line in out.split("\n") if line.startswith("*"))
        current.replace("* ", "")
    except subprocess.CalledProcessError:
        current = "inside_docker"
    except (FileNotFoundError, StopIteration) as e:
        current = "unknown"
    return current


def to_camel(text):
    text = text.capitalize()
    text = re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)
    text = text.replace("Tts", "TTS")
    text = text.replace("vc", "VC")
    return text


def find_module(module_path: str, module_name: str) -> object:
    module_name = module_name.lower()
    module = importlib.import_module(module_path + "." + module_name)
    class_name = to_camel(module_name)
    return getattr(module, class_name)


def import_class(module_path: str) -> object:
    """Import a class from a module path.

    Args:
        module_path (str): The module path of the class.

    Returns:
        object: The imported class.
    """
    class_name = module_path.split(".")[-1]
    module_path = ".".join(module_path.split(".")[:-1])
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_import_path(obj: object) -> str:
    """Get the import path of a class.

    Args:
        obj (object): The class object.

    Returns:
        str: The import path of the class.
    """
    return ".".join([type(obj).__module__, type(obj).__name__])


def get_user_data_dir(appname):
    TTS_HOME = os.environ.get("TTS_HOME")
    XDG_DATA_HOME = os.environ.get("XDG_DATA_HOME")
    if TTS_HOME is not None:
        ans = Path(TTS_HOME).expanduser().resolve(strict=False)
    elif XDG_DATA_HOME is not None:
        ans = Path(XDG_DATA_HOME).expanduser().resolve(strict=False)
    elif sys.platform == "win32":
        import winreg  # pylint: disable=import-outside-toplevel

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
        )
        dir_, _ = winreg.QueryValueEx(key, "Local AppData")
        ans = Path(dir_).resolve(strict=False)
    elif sys.platform == "darwin":
        ans = Path("~/Library/Application Support/").expanduser()
    else:
        ans = Path.home().joinpath(".local/share")
    return ans.joinpath(appname)


def set_init_dict(model_dict, checkpoint_state, c):
    # Partial initialization: if there is a mismatch with new and old layer, it is skipped.
    for k, v in checkpoint_state.items():
        if k not in model_dict:
            logger.warning("Layer missing in the model finition %s", k)
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in checkpoint_state.items() if k in model_dict}
    # 2. filter out different size layers
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if v.numel() == model_dict[k].numel()}
    # 3. skip reinit layers
    if c.has("reinit_layers") and c.reinit_layers is not None:
        for reinit_layer_name in c.reinit_layers:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if reinit_layer_name not in k}
    # 4. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    logger.info("%d / %d layers are restored.", len(pretrained_dict), len(model_dict))
    return model_dict


def format_aux_input(def_args: Dict, kwargs: Dict) -> Dict:
    """Format kwargs to hande auxilary inputs to models.

    Args:
        def_args (Dict): A dictionary of argument names and their default values if not defined in `kwargs`.
        kwargs (Dict): A `dict` or `kwargs` that includes auxilary inputs to the model.

    Returns:
        Dict: arguments with formatted auxilary inputs.
    """
    kwargs = kwargs.copy()
    for name in def_args:
        if name not in kwargs or kwargs[name] is None:
            kwargs[name] = def_args[name]
    return kwargs


def get_timestamp() -> str:
    return datetime.now().strftime("%y%m%d-%H%M%S")


class ConsoleFormatter(logging.Formatter):
    """Custom formatter that prints logging.INFO messages without the level name.

    Source: https://stackoverflow.com/a/62488520
    """

    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = "%(message)s"
        else:
            self._style._fmt = "%(levelname)s: %(message)s"
        return super().format(record)


def setup_logger(
    logger_name: str,
    level: int = logging.INFO,
    *,
    formatter: Optional[logging.Formatter] = None,
    screen: bool = False,
    tofile: bool = False,
    log_dir: str = "logs",
    log_name: str = "log",
) -> None:
    lg = logging.getLogger(logger_name)
    if formatter is None:
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d - %(levelname)-8s - %(name)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S"
        )
    lg.setLevel(level)
    if tofile:
        Path(log_dir).mkdir(exist_ok=True, parents=True)
        log_file = Path(log_dir) / f"{log_name}_{get_timestamp()}.log"
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
