import torch
import re

from tokenizers import Tokenizer
from functools import cached_property
from num2words import num2words
from tts.xtts.zh_num2words import TextNorm as zh_num2words


class VoiceBpeTokenizer:
    def __init__(self, vocab_file):
        self.whitespace_re = re.compile(r"\s+")

        self.tokenizer = None
        self.tokenizer = Tokenizer.from_file(vocab_file)

        self.char_limits = {
            "en": 250,
            "de": 253,
            "fr": 273,
            "es": 239,
            "it": 213,
            "pt": 203,
            "pl": 224,
            "zh": 82,
            "ar": 166,
            "cs": 186,
            "ru": 182,
            "nl": 251,
            "tr": 226,
            "ja": 71,
            "hu": 224,
            "ko": 95,
            "hi": 150,
            "uk": 182,
        }

        self.ordinal_re = {
            "en": re.compile(r"([0-9]+)(st|nd|rd|th)"),
            "es": re.compile(r"([0-9]+)(º|ª|er|o|a|os|as)"),
            "fr": re.compile(r"([0-9]+)(º|ª|er|re|e|ème)"),
            "de": re.compile(r"([0-9]+)(st|nd|rd|th|º|ª|\.(?=\s|$))"),
            "pt": re.compile(r"([0-9]+)(º|ª|o|a|os|as)"),
            "it": re.compile(r"([0-9]+)(º|°|ª|o|a|i|e)"),
            "pl": re.compile(r"([0-9]+)(º|ª|st|nd|rd|th)"),
            "ar": re.compile(r"([0-9]+)(ون|ين|ث|ر|ى)"),
            "cs": re.compile(r"([0-9]+)\.(?=\s|$)"),
            # In Czech, a dot is often used after the number to indicate ordinals.
            "ru": re.compile(r"([0-9]+)(-й|-я|-е|-ое|-ье|-го)"),
            "nl": re.compile(r"([0-9]+)(de|ste|e)"),
            "tr": re.compile(r"([0-9]+)(\.|inci|nci|uncu|üncü|\.)"),
            "hu": re.compile(r"([0-9]+)(\.|adik|edik|odik|edik|ödik|ödike|ik)"),
            "ko": re.compile(r"([0-9]+)(번째|번|차|째)"),
        }
        self.number_re = re.compile(r"[0-9]+")
        self.currency_re = {
            "USD": re.compile(r"((\$[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+\$))"),
            "GBP": re.compile(r"((£[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+£))"),
            "EUR": re.compile(r"(([0-9\.\,]*[0-9]+€)|((€[0-9\.\,]*[0-9]+)))"),
        }

        self.comma_number_re = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")
        self.dot_number_re = re.compile(r"\b\d{1,3}(.\d{3})*(\,\d+)?\b")
        self.decimal_number_re = re.compile(r"([0-9]+[.,][0-9]+)")

        self.symbols_multilingual = {
            "en": [
                (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
                for x in [
                    ("&", " and "),
                    ("@", " at "),
                    ("%", " percent "),
                    ("#", " hash "),
                    ("$", " dollar "),
                    ("£", " pound "),
                    ("°", " degree "),
                ]
            ],
            "es": [
                (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
                for x in [
                    ("&", " y "),
                    ("@", " arroba "),
                    ("%", " por ciento "),
                    ("#", " numeral "),
                    ("$", " dolar "),
                    ("£", " libra "),
                    ("°", " grados "),
                ]
            ],
            "fr": [
                (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
                for x in [
                    ("&", " et "),
                    ("@", " arobase "),
                    ("%", " pour cent "),
                    ("#", " dièse "),
                    ("$", " dollar "),
                    ("£", " livre "),
                    ("°", " degrés "),
                ]
            ],
            "de": [
                (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
                for x in [
                    ("&", " und "),
                    ("@", " at "),
                    ("%", " prozent "),
                    ("#", " raute "),
                    ("$", " dollar "),
                    ("£", " pfund "),
                    ("°", " grad "),
                ]
            ],
            "pt": [
                (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
                for x in [
                    ("&", " e "),
                    ("@", " arroba "),
                    ("%", " por cento "),
                    ("#", " cardinal "),
                    ("$", " dólar "),
                    ("£", " libra "),
                    ("°", " graus "),
                ]
            ],
            "it": [
                (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
                for x in [
                    ("&", " e "),
                    ("@", " chiocciola "),
                    ("%", " per cento "),
                    ("#", " cancelletto "),
                    ("$", " dollaro "),
                    ("£", " sterlina "),
                    ("°", " gradi "),
                ]
            ],
            "pl": [
                (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
                for x in [
                    ("&", " i "),
                    ("@", " małpa "),
                    ("%", " procent "),
                    ("#", " krzyżyk "),
                    ("$", " dolar "),
                    ("£", " funt "),
                    ("°", " stopnie "),
                ]
            ],
            "ar": [
                # Arabic
                (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
                for x in [
                    ("&", " و "),
                    ("@", " على "),
                    ("%", " في المئة "),
                    ("#", " رقم "),
                    ("$", " دولار "),
                    ("£", " جنيه "),
                    ("°", " درجة "),
                ]
            ],
            "zh": [
                # Chinese
                (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
                for x in [
                    ("&", " 和 "),
                    ("@", " 在 "),
                    ("%", " 百分之 "),
                    ("#", " 号 "),
                    ("$", " 美元 "),
                    ("£", " 英镑 "),
                    ("°", " 度 "),
                ]
            ],
            "cs": [
                # Czech
                (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
                for x in [
                    ("&", " a "),
                    ("@", " na "),
                    ("%", " procento "),
                    ("#", " křížek "),
                    ("$", " dolar "),
                    ("£", " libra "),
                    ("°", " stupně "),
                ]
            ],
            "ru": [
                # Russian
                (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
                for x in [
                    ("&", " и "),
                    ("@", " собака "),
                    ("%", " процентов "),
                    ("#", " номер "),
                    ("$", " доллар "),
                    ("£", " фунт "),
                    ("°", " градус "),
                ]
            ],
            "nl": [
                # Dutch
                (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
                for x in [
                    ("&", " en "),
                    ("@", " bij "),
                    ("%", " procent "),
                    ("#", " hekje "),
                    ("$", " dollar "),
                    ("£", " pond "),
                    ("°", " graden "),
                ]
            ],
            "tr": [
                (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
                for x in [
                    ("&", " ve "),
                    ("@", " at "),
                    ("%", " yüzde "),
                    ("#", " diyez "),
                    ("$", " dolar "),
                    ("£", " sterlin "),
                    ("°", " derece "),
                ]
            ],
            "hu": [
                (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
                for x in [
                    ("&", " és "),
                    ("@", " kukac "),
                    ("%", " százalék "),
                    ("#", " kettőskereszt "),
                    ("$", " dollár "),
                    ("£", " font "),
                    ("°", " fok "),
                ]
            ],
            "ko": [
                # Korean
                (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
                for x in [
                    ("&", " 그리고 "),
                    ("@", " 에 "),
                    ("%", " 퍼센트 "),
                    ("#", " 번호 "),
                    ("$", " 달러 "),
                    ("£", " 파운드 "),
                    ("°", " 도 "),
                ]
            ],
        }
        self.abbreviations = {
            "en": [
                (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
                for x in [
                    ("mrs", "misess"),
                    ("mr", "mister"),
                    ("dr", "doctor"),
                    ("st", "saint"),
                    ("co", "company"),
                    ("jr", "junior"),
                    ("maj", "major"),
                    ("gen", "general"),
                    ("drs", "doctors"),
                    ("rev", "reverend"),
                    ("lt", "lieutenant"),
                    ("hon", "honorable"),
                    ("sgt", "sergeant"),
                    ("capt", "captain"),
                    ("esq", "esquire"),
                    ("ltd", "limited"),
                    ("col", "colonel"),
                    ("ft", "fort"),
                ]
            ],
            "es": [
                (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
                for x in [
                    ("sra", "señora"),
                    ("sr", "señor"),
                    ("dr", "doctor"),
                    ("dra", "doctora"),
                    ("st", "santo"),
                    ("co", "compañía"),
                    ("jr", "junior"),
                    ("ltd", "limitada"),
                ]
            ],
            "fr": [
                (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
                for x in [
                    ("mme", "madame"),
                    ("mr", "monsieur"),
                    ("dr", "docteur"),
                    ("st", "saint"),
                    ("co", "compagnie"),
                    ("jr", "junior"),
                    ("ltd", "limitée"),
                ]
            ],
            "de": [
                (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
                for x in [
                    ("fr", "frau"),
                    ("dr", "doktor"),
                    ("st", "sankt"),
                    ("co", "firma"),
                    ("jr", "junior"),
                ]
            ],
            "pt": [
                (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
                for x in [
                    ("sra", "senhora"),
                    ("sr", "senhor"),
                    ("dr", "doutor"),
                    ("dra", "doutora"),
                    ("st", "santo"),
                    ("co", "companhia"),
                    ("jr", "júnior"),
                    ("ltd", "limitada"),
                ]
            ],
            "it": [
                (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
                for x in [
                    # ("sig.ra", "signora"),
                    ("sig", "signore"),
                    ("dr", "dottore"),
                    ("st", "santo"),
                    ("co", "compagnia"),
                    ("jr", "junior"),
                    ("ltd", "limitata"),
                ]
            ],
            "pl": [
                (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
                for x in [
                    ("p", "pani"),
                    ("m", "pan"),
                    ("dr", "doktor"),
                    ("sw", "święty"),
                    ("jr", "junior"),
                ]
            ],
            "ar": [
                (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
                for x in [
                    # There are not many common abbreviations in Arabic as in English.
                ]
            ],
            "zh": [
                (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
                for x in [
                    # Chinese doesn't typically use abbreviations in the same way as Latin-based scripts.
                ]
            ],
            "cs": [
                (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
                for x in [
                    ("dr", "doktor"),  # doctor
                    ("ing", "inženýr"),  # engineer
                    ("p", "pan"),  # Could also map to pani for woman but no easy way to do it
                    # Other abbreviations would be specialized and not as common.
                ]
            ],
            "ru": [
                (re.compile("\\b%s\\b" % x[0], re.IGNORECASE), x[1])
                for x in [
                    ("г-жа", "госпожа"),  # Mrs.
                    ("г-н", "господин"),  # Mr.
                    ("д-р", "доктор"),  # doctor
                    # Other abbreviations are less common or specialized.
                ]
            ],
            "nl": [
                (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
                for x in [
                    ("dhr", "de heer"),  # Mr.
                    ("mevr", "mevrouw"),  # Mrs.
                    ("dr", "dokter"),  # doctor
                    ("jhr", "jonkheer"),  # young lord or nobleman
                    # Dutch uses more abbreviations, but these are the most common ones.
                ]
            ],
            "tr": [
                (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
                for x in [
                    ("b", "bay"),  # Mr.
                    ("byk", "büyük"),  # büyük
                    ("dr", "doktor"),  # doctor
                    # Add other Turkish abbreviations here if needed.
                ]
            ],
            "hu": [
                (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
                for x in [
                    ("dr", "doktor"),  # doctor
                    ("b", "bácsi"),  # Mr.
                    ("nőv", "nővér"),  # nurse
                    # Add other Hungarian abbreviations here if needed.
                ]
            ],
            "ko": [
                (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
                for x in [
                    # Korean doesn't typically use abbreviations in the same way as Latin-based scripts.
                ]
            ],
        }

    @cached_property
    def katsu(self):
        import cutlet

        return cutlet.Cutlet()

    def check_input_length(self, txt, lang):
        lang = lang.split("-")[0]  # remove the region
        limit = self.char_limits.get(lang, 250)
        if len(txt) > limit:
            print(f"The text length exceeds the character limit of {limit} for language '{lang}', this might cause truncated audio.")

    def preprocess_text(self, txt, lang):
        if lang in {"ar", "cs", "de", "en", "es", "fr", "hu", "it", "nl", "pl", "pt", "ru", "tr", "zh", "ko"}:
            txt = self.multilingual_cleaners(txt, lang)
            if lang == "zh":
                txt = self.chinese_transliterate(txt)
            if lang == "ko":
                txt = self.korean_transliterate(txt)
        elif lang == "ja":
            txt = self.japanese_cleaners(txt, self.katsu)
        elif lang == "hi":
            # @manmay will implement this
            txt = self.basic_cleaners(txt)
        else:
            raise NotImplementedError(f"Language '{lang}' is not supported.")
        return txt

    def encode(self, txt, lang):
        lang = lang.split("-")[0]  # remove the region
        self.check_input_length(txt, lang)
        txt = self.preprocess_text(txt, lang)
        lang = "zh-cn" if lang == "zh" else lang
        txt = f"[{lang}]{txt}"
        txt = txt.replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", "")
        txt = txt.replace("[UNK]", "")
        return txt

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def get_number_tokens(self):
        return max(self.tokenizer.get_vocab().values()) + 1

    def lowercase(self, text):
        return text.lower()

    def collapse_whitespace(self, text):
        return re.sub(self.whitespace_re, " ", text)

    def multilingual_cleaners(self, text, lang):
        text = text.replace('"', "")
        if lang == "tr":
            text = text.replace("İ", "i")
            text = text.replace("Ö", "ö")
            text = text.replace("Ü", "ü")
        text = self.lowercase(text)
        text = self.expand_numbers_multilingual(text, lang)
        text = self.expand_abbreviations_multilingual(text, lang)
        text = self.expand_symbols_multilingual(text, lang=lang)
        text = self.collapse_whitespace(text)
        return text

    def basic_cleaners(self, text):
        """Basic pipeline that lowercases and collapses whitespace without transliteration."""
        text = self.lowercase(text)
        text = self.collapse_whitespace(text)
        return text

    def chinese_transliterate(self, text):
        try:
            import pypinyin
        except ImportError as e:
            raise ImportError("Chinese requires: pypinyin") from e
        return "".join(
            [p[0] for p in
             pypinyin.pinyin(text, style=pypinyin.Style.TONE3, heteronym=False, neutral_tone_with_five=True)]
        )

    def japanese_cleaners(self, text, katsu):
        text = katsu.romaji(text)
        text = self.lowercase(text)
        return text

    def korean_transliterate(self, text):
        try:
            from hangul_romanize import Transliter
            from hangul_romanize.rule import academic
        except ImportError as e:
            raise ImportError("Korean requires: hangul_romanize") from e
        r = Transliter(academic)
        return r.translit(text)

    def expand_symbols_multilingual(self, text, lang="en"):
        for regex, replacement in self.symbols_multilingual[lang]:
            text = re.sub(regex, replacement, text)
            text = text.replace("  ", " ")  # Ensure there are no double spaces
        return text.strip()

    def expand_abbreviations_multilingual(self, text, lang="en"):
        for regex, replacement in self.abbreviations[lang]:
            text = re.sub(regex, replacement, text)
        return text

    def expand_numbers_multilingual(self, text, lang="en"):
        if lang == "zh":
            text = zh_num2words()(text)
        else:
            if lang in ["en", "ru"]:
                text = re.sub(self.comma_number_re, self.remove_commas, text)
            else:
                text = re.sub(self.dot_number_re, self.remove_dots, text)
            try:
                text = re.sub(self.currency_re["GBP"], lambda m: self.expand_currency(m, lang, "GBP"), text)
                text = re.sub(self.currency_re["USD"], lambda m: self.expand_currency(m, lang, "USD"), text)
                text = re.sub(self.currency_re["EUR"], lambda m: self.expand_currency(m, lang, "EUR"), text)
            except:
                pass
            if lang != "tr":
                text = re.sub(self.decimal_number_re, lambda m: self.expand_decimal_point(m, lang), text)
            text = re.sub(self.ordinal_re[lang], lambda m: self.expand_ordinal(m, lang), text)
            text = re.sub(self.number_re, lambda m: self.expand_number(m, lang), text)
        return text

    def remove_commas(self, m):
        text = m.group(0)
        if "," in text:
            text = text.replace(",", "")
        return text

    def remove_dots(self, m):
        text = m.group(0)
        if "." in text:
            text = text.replace(".", "")
        return text

    def expand_decimal_point(self, m, lang="en"):
        amount = m.group(1).replace(",", ".")
        return num2words(float(amount), lang=lang if lang != "cs" else "cz")

    def expand_currency(self, m, lang="en", currency="USD"):
        amount = float((re.sub(r"[^\d.]", "", m.group(0).replace(",", "."))))
        full_amount = num2words(amount, to="currency", currency=currency, lang=lang if lang != "cs" else "cz")

    def expand_ordinal(self, m, lang="en"):
        return num2words(int(m.group(1)), ordinal=True, lang=lang if lang != "cs" else "cz")

    def expand_number(self, m, lang="en"):
        return num2words(int(m.group(0)), lang=lang if lang != "cs" else "cz")
