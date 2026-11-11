"""Finnish language parameters for the FUTO Keyboard LM pipeline.

Constants below are pinned to the verified reference spec of FUTO's English
keyboard model (breadlicker45/futo-keyboard-lm on HuggingFace), as documented
in the futo-portuguese project's GUIDE.md. The FUTO C++ loader validates a
several of these values, so they must not be changed casually.

The 300 user-defined SentencePiece symbols occupy token ids 4..303 in
declaration order. Only the content slots are language-specific; everything
else (filler, structural, CHAR, emoji slots) is kept byte-identical in layout
to the reference model.
"""
from __future__ import annotations

# ---------------------------------------------------------------- reference --

# BCP-47 tag stored in keyboardlm.languages.
LANG_TAG = "fi"

# Model history line embedded into the GGUF.
HISTORY_TEMPLATE = "{date}: FUTO Finnish keyboard LM v{version} (36M Llama, {mode} mode)"

# Vanilla Llama config, byte-compatible with the reference English model.
LLAMA_CONFIG = dict(
    vocab_size=15008,
    hidden_size=512,
    intermediate_size=1024,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=8,          # MHA, no GQA
    max_position_embeddings=2048,    # NOT 512 (wiki was wrong; reference is 2048)
    rms_norm_eps=1e-6,              # NOT 1e-5 (wiki was wrong)
    rope_theta=10_000.0,
    tie_word_embeddings=False,
)

VOCAB_SIZE = 15008

# Feature flags the FUTO Android app requires. char_embed_mixing_v1 is NOT
# optional when xbu_char_autocorrect_v1 is declared: without it the C++ does a
# nullptr deref on every keypress (verified crash in the pt-BR project).
GGUF_FEATURES = "base_v1 inverted_space xbu_char_autocorrect_v1 char_embed_mixing_v1"

# HF repo of the official English reference model (used as the spec source and
# for the final metadata diff + smoke test).
REFERENCE_HF_REPO = "breadlicker45/futo-keyboard-lm"
REFERENCE_GGUF_FILENAME = "ml4_1_f16_meta_fixed.gguf"

# ------------------------------------------------- user-defined symbol slots --

# Slot 174..176: autocorrect structure.
XBU, XBC, XEC = "<XBU>", "<XBC>", "<XEC>"
# Slot 177..181: only <XC0> is referenced by the app; keep the rest in place.
XC_RESERVED = ["<XC0>", "<XC1>", "<XC2>", "<XC3>", "<XC4>"]
# Slots 182..207: one token per physical keypress A..Z. The C++ resolves
# <CHAR_A> by name and computes the rest as LETTERS_TO_IDS[0] + i, so these
# MUST be 26 contiguous, sequential ids in the final vocabulary.
CHAR_TOKENS = [f"<CHAR_{c}>" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]

# Content slots 28..173 (146 slots): high-frequency Finnish words. Pinning them
# as single BPE pieces makes both next-word output and autocorrect cheap for
# the most common words.
FI_CONTENT_A = [
    "ja", "ei", "se", "että", "on", "oli", "kun", "joka", "niin", "mutta",
    "tai", "koska", "mitä", "missä", "milloin", "miten", "miksi", "kuka",
    "kenet", "kenen", "paljon", "vähän", "nyt", "tänään", "huomenna", "eilen",
    "aamulla", "iltapäivällä", "illalla", "yöllä", "päivällä", "päivä",
    "vuosi", "kuukausi", "viikko", "sunnuntai", "maanantai", "tiistai",
    "keskiviikko", "torstai", "perjantai", "lauantai", "tammikuu",
    "helmikuu", "maaliskuu", "huhtikuu", "toukokuu", "kesäkuu", "heinäkuu",
    "elokuu", "syyskuu", "lokakuu", "marraskuu", "joulukuu", "hei", "moi",
    "moikka", "kiitos", "tervetuloa", "hyvä", "huono", "kyllä", "todella",
    "varmasti", "tietysti", "ehkä", "näin", "täällä", "siellä", "kotona",
    "työssä", "koulussa", "me", "te", "minä", "sinä", "hän", "tämä", "tuo",
    "meidän", "teidän", "heidän", "oma", "omaa", "isä", "äiti", "veli",
    "sisar", "perhe", "ystävä", "kaveri", "koulu", "työ", "raha", "aika",
    "elämä", "maa", "koti", "talo", "huone", "ovi", "ikkuna", "pöytä",
    "tuoli", "sänky", "syy", "sata", "tuhat", "miljoona", "voi", "joo",
    "anteeksi", "hyvästi", "kysymys", "vastaus", "apu", "tieto", "asia",
    "juttu", "tapahtuma", "tilanne", "tila", "paikka", "matka", "ajaa",
    "lähteä", "tulla", "päästä", "pois", "portti", "avata", "sulkea",
    "pieni", "suuri", "uusi", "vanha", "lapsi", "lapsen", "suku", "sisko",
    "nainen", "mies", "ihminen", "ihmisiä", "hyvää", "kaikki",
]

# Content slots 208..263 (56 slots): a second tier of frequent verbs/nouns.
FI_CONTENT_B = [
    "kysy", "kysyy", "kysyi", "vastaa", "auttaa", "syö", "syödä", "juo",
    "juoda", "nuku", "nukuta", "juokse", "juosta", "kävele", "kävellä",
    "puhu", "puhua", "kirjoita", "kirjoittaa", "lukea", "lukee", "nähdä",
    "näkee", "kuulla", "kuulee", "tuntea", "ajatella", "tietää", "uskoa",
    "rakastaa", "vihaa", "pelkää", "pelätä", "odottaa", "odota", "etsiä",
    "löytää", "anna", "ottaa", "laita", "vie", "tuli", "mene", "tule",
    "pysy", "istu", "nauraa", "itkeä", "valittaa", "kiittää",
    "pyytää", "tarjota", "ostaa", "myydä", "maksaa", "säästää",
]

# Sanity: these must line up with the slot map or the model layout breaks.
assert len(FI_CONTENT_A) == 146, f"FI_CONTENT_A must have 146 entries, has {len(FI_CONTENT_A)}"
assert len(FI_CONTENT_B) == 56, f"FI_CONTENT_B must have 56 entries, has {len(FI_CONTENT_B)}"
assert len(set(FI_CONTENT_A + FI_CONTENT_B)) == len(FI_CONTENT_A + FI_CONTENT_B), "duplicate content words"


def build_user_symbols(ref_pieces: list[str]) -> list[str]:
    """Assemble the 300 user-defined symbols in declaration order.

    ref_pieces[i] is piece i of the reference (English) SentencePiece model.
    Filler (4..27), structural (174..181), CHAR (182..207) and emoji (264..303)
    slots are kept from the reference for layout parity; the two content blocks
    are replaced with Finnish.
    """
    assert len(ref_pieces) == VOCAB_SIZE, f"reference vocab {len(ref_pieces)} != {VOCAB_SIZE}"
    symbols = []
    symbols += ref_pieces[4:28]                 # <FUTO0>..<FUTO23> filler
    symbols += FI_CONTENT_A                     # content slots 28..173
    symbols += [XBU, XBC, XEC] + XC_RESERVED    # 174..181
    symbols += CHAR_TOKENS                      # 182..207
    symbols += FI_CONTENT_B                     # 208..263
    symbols += ref_pieces[264:304]             # emoji 264..303
    assert len(symbols) == 300
    return symbols


# ------------------------------------------------------- Finnish typo model --

# Finnish QWERTY layout: rows of keys. Å/Ö/Ä replace the QWERTY right edge.
FINNISH_LAYOUT_ROWS = [
    list("1234567890=-"),
    list("QWERTYUIOPÅ"),
    list("ASDFGHJKLÖÄ"),
    list("ZXCVBNM"),
]

# Adjacent-key map for synthetic typos (horizontal neighbours plus the
# half-key-stagger above/below). Keys map to the base character they type.
KEY_ADJACENCY = {
    "Q": "WA", "W": "QESA", "E": "WRSD", "R": "ETDF", "T": "RYFG",
    "Y": "TUGH", "U": "YIHK", "I": "UOJK", "O": "IPKL", "P": "OÅLÖ", "Å": "PÖ",
    "A": "QWSZX", "S": "AWEDZX", "D": "SERFCX", "F": "RTGCV", "G": "TYHVB",
    "H": "YUJBN", "J": "UIKNM", "K": "IJOLM", "L": "KOPÖÄ",
    "Ö": "LKÄÅP", "Ä": "LÖ",
    "Z": "SX", "X": "SZDC", "C": "DXFV", "V": "FGCB", "B": "GHVN",
    "N": "HJMB", "M": "NB",
}

# Diacritic drop is the dominant Finnish typo class: the physical key is a
# base letter and the diacritic is not a separate keypress.
DIACRITIC_BASE = {"ä": "a", "ö": "o", "å": "a"}


def typed_to_keypress_chars(typed: str) -> list[str]:
    """Map a typed string to the <CHAR_*> token sequence the keyboard sends.

    One token per physical keypress; diacritics collapse to their base key.
    """
    out = []
    for ch in typed.lower():
        base = DIACRITIC_BASE.get(ch, ch)
        if "a" <= base <= "z":
            out.append(f"<CHAR_{base.upper()}>")
    return out


# ------------------------------------------------------------- eval suite ----

# Plausible Finnish typos -> correct word, for the autocorrect eval.
# context is optional preceding text; the prompt is
#   [context + " "] + <XBU> + keypress_chars(typo) + <XBC>
# and the model must emit the corrected word before <XEC>.
EVAL_SUITE = [
    dict(typo="hyva", word="hyvä"),
    dict(typo="kytos", word="kiitos"),
    dict(typo="tma", word="tämä"),
    dict(typo="msta", word="mistä"),
    dict(typo="oln", word="olen"),
    dict(typo="kyll", word="kyllä"),
    dict(typo="ettko", word="etkö"),
    dict(typo="tssa", word="tässä"),
    dict(typo="koton", word="kotona"),
    dict(typo="tyssa", word="työssä"),
    dict(typo="koulus", word="koulussa"),
    dict(typo="pajon", word="paljon"),
    dict(typo="vhaan", word="vähän"),
    dict(typo="isoo", word="iso"),
    dict(typo="kaikk", word="kaikki"),
    dict(typo="koskaa", word="koskaan"),
    dict(typo="mika", word="mikä"),
    dict(typo="enaa", word="enää"),
    dict(typo="saa", word="sää"),
    dict(typo="mitaa", word="mitää"),
    dict(typo="onkko", word="onko", context="Eikö"),
    dict(typo="kysy", word="kyse", context="Ei ole"),
    dict(typo="tervetullaa", word="tervetuloa"),
    dict(typo="hyyn", word="hyvin"),
    dict(typo="niik", word="niin", context="Miten"),
    dict(typo="msta", word="mistä", context="Missä"),
    dict(typo="hien", word="hienoa"),
    dict(typo="hieno", word="hienoa"),
    dict(typo="terve", word="tervetuloa", context="Moi"),
    dict(typo="moi", word="moi"),
]

# ------------------------------------------------------------ training data --

# HF corpus sources, in mix order. All streamable via the datasets library.
# oscar: bulk cleaned web crawl; opus-100 fi: movie subtitles (casual register,
# the strongest next-word signal for typing); wiki: formal grounding.
CORPUS_SOURCES = [
    dict(
        name="oscar_fi",
        dataset="Finnish-NLP/oscar_2301_fi_cleaned",
        split="train",
        column="text",
        weight=0.50,
        min_chars=60,
        max_chars=20_000,
        filter_finnish=True,
    ),
    dict(
        name="opus_fi",
        dataset="Helsinki-NLP/opus-100",
        config="en-fi",
        split="train",
        column="translation",   # en-fi config: source=en, translation=fi
        # datasets 3.x rows for a pair config are translation={en, fi}; pick the fi side.
        subcolumn="fi",
        weight=0.35,
        min_chars=4,
        max_chars=500,
        filter_finnish=False,    # parallel fi side; trust the source
    ),
    dict(
        name="wiki_fi",
        dataset="wikimedia/wikipedia",
        config="20231101.fi",
        split="train",
        column="text",
        weight=0.15,
        min_chars=80,
        max_chars=20_000,
        filter_finnish=True,
    ),
]

# Rough Finnish BPE chars-per-token estimate for budgeting.
CHARS_PER_TOKEN = 3.8

# ------------------------------------------------------- training recipes ----

PRETRAIN = dict(
    seq_len=1024,
    micro_batch=16,
    grad_accum=8,           # effective batch 128 -> ~131k tokens/step
    learning_rate=3e-4,
    warmup_steps=2000,
    weight_decay=0.1,
    save_steps=5000,
)

# Three-stage autocorrect finetune (see GUIDE.md phase 4).
FINETUNE = dict(
    # 4a: isolated <XBU>...<XBC>word<XEC> triples, loss masked to the correction span
    a=dict(seq_len=64, batch=256, learning_rate=1e-4, typo_variants_per_word=4),
    # 4b: real sentences with 25% of words replaced by typo triples, full labels
    b=dict(seq_len=512, batch=64, learning_rate=5e-5, typo_rate=0.25, sentences=20_000),
    # 4c: casual register (subtitles) with a lower typo rate, short polish step
    c=dict(seq_len=512, batch=64, learning_rate=2e-5, typo_rate=0.10, sentences=20_000),
)

# Per-mode step counts / data budgets.
MODES = {
    "mini": dict(
        pretrain_steps=20_000,
        token_budget=500_000_000,      # ~1.9 GB of text
        ft_a_steps=5_000,
        ft_b_steps=10_000,
        ft_c_steps=5_000,
    ),
    "full": dict(
        pretrain_steps=100_000,
        token_budget=2_500_000_000,    # ~9.5 GB of text
        ft_a_steps=10_000,
        ft_b_steps=20_000,
        ft_c_steps=5_000,
    ),
}

# SentencePiece training sample cap (GB): BPE training is RAM-bound (~6-10x the
# sample size), so cap the sample regardless of corpus size.
SPM_SAMPLE_GB = 1.5

# Reference-model smoke test prompts used at export time.
SMOKE_PROMPTS = [
    "Moi <XBU>hii<XBC>",
    "Terhde <XBU>thetull<XBC>",
    "Moi, mä olen <XBU>juho<XBC>",
]
