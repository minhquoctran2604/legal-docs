"""Central configuration for the legal_ner pipeline.

Paths, crawl settings and the NER label inventory live here so every CLI
shares the same defaults.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"          # downloaded PDFs / HTML + crawl metadata
TEXT_DIR = DATA_DIR / "text"        # normalized plain-text judgments
LABELED_DIR = DATA_DIR / "labeled"  # weak-labeled JSONL (BIO)
MODELS_DIR = DATA_DIR / "models"    # fine-tuned checkpoints

# ---------------------------------------------------------------------------
# Crawler settings (congbobanan.toaan.gov.vn)
# ---------------------------------------------------------------------------
PORTAL_BASE = "https://congbobanan.toaan.gov.vn"
DETAIL_URL = PORTAL_BASE + "/2ta{id}t1cvn/chi-tiet-ban-an"
FULLTEXT_URL = PORTAL_BASE + "/3ta{id}t1cvn"  # serves the judgment PDF bytes
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
CRAWL_DELAY_SECONDS = 1.0      # polite delay between requests
REQUEST_TIMEOUT = 30           # seconds
DEFAULT_START_ID = 100000      # verified criminal-case region of the id space
CRAWL_STATE_FILE = RAW_DIR / "crawl_state.json"

# Substrings (lowercased) identifying criminal cases in the detail-page title.
CRIMINAL_MARKERS = ("hình sự", "hs-st", "hs-pt", "hsst", "hspt")

# Substrings (lowercased) identifying civil/family/administrative cases in the
# detail-page title. Two complementary signal families:
#   1. case-number type suffixes  -> "08/2018/QĐDS-ST", "31/2018/HCPT", ...
#        DS   = dan su (civil)            HNGD = hon nhan gia dinh (family)
#        KDTM = kinh doanh thuong mai     LD   = lao dong (labour)
#        HC   = hanh chinh (administrative)
#   2. topic phrases for the many older titles that carry no type suffix
#        (e.g. "số 124 ngày ... Vụ án ly hôn ...").
# A judgment counts as civil only when it matches one of these AND is NOT
# criminal -> this rejects criminal titles that lack an explicit HS suffix.
CIVIL_MARKERS = (
    # --- case-number type suffixes (DS / HNGĐ / KDTM / LĐ / HC) ---
    "ds-st", "ds-pt", "dsst", "dspt", "qđds", "qдds",
    "hngđ", "hngd", "hn-st", "hn-pt", "hnst", "hnpt",
    "kdtm", "kdtm-st", "kdtm-pt",
    "lđ-st", "lđ-pt", "lđst", "lđpt", "ld-st", "ld-pt",
    "hc-st", "hc-pt", "hcst", "hcpt", "qđhc",
    # --- explicit case-type words ---
    "dân sự", "hôn nhân", "gia đình", "hành chính",
    "kinh doanh", "thương mại", "lao động",
    # --- common civil/family/admin topic phrases (suffix-less titles) ---
    "ly hôn", "tranh chấp", "khiếu kiện", "yêu cầu",
    "chia tài sản", "thừa kế", "đòi nợ", "hợp đồng",
)

# ---------------------------------------------------------------------------
# NER label inventory (BIO scheme)
# ---------------------------------------------------------------------------
# 20 entity types -> 41 BIO tags (O + B-/I- per type).
ENTITY_TYPES = [
    # A. document metadata
    "CASE_NUMBER",    # judgment/decision number: "17/2018/HS-ST"
    "COURT",          # court name
    "JUDGMENT_DATE",  # header / hearing / pronouncement date
    "CASE_TYPE",      # explicit case-type mention: "hình sự", "dân sự", ...
    # B. parties (anonymized names after role anchors)
    "DEFENDANT",      # name after "bị cáo / các bị cáo"
    "PLAINTIFF",      # name after "nguyên đơn"
    "VICTIM",         # name after "bị hại / người bị hại"
    "RELATED_PARTY",  # name after "người có quyền lợi, nghĩa vụ liên quan"
    # C. legal references
    "LAW_NAME",       # "Bộ luật Hình sự năm 2015 (sửa đổi, bổ sung năm 2017)"
    "ARTICLE",        # "Điều 250"
    "CLAUSE",         # "khoản 1"
    "POINT",          # "điểm c"
    "LEGAL_BASIS",    # full "Căn cứ ..."/"Áp dụng ..." citation sentences
    # D. offense
    "CRIME",          # offense name (quoted or unquoted)
    "VIOLATION_ACT",  # violating-act phrase after "có hành vi ..." (fuzzy)
    # E. outcome
    "PENALTY",        # "02 (hai) năm tù", "tù chung thân", "án treo", ...
    "MONEY_AMOUNT",   # money NOT in compensation/fee context
    "COMPENSATION",   # money in "bồi thường" context
    "COURT_FEE",      # money in "án phí / lệ phí" context
    "DECISION",       # verdict-sentence remainder in QUYET DINH section
]

LABELS = ["O"]
for _ent in ENTITY_TYPES:
    LABELS.append(f"B-{_ent}")
    LABELS.append(f"I-{_ent}")

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
MODEL_NAME = "xlm-roberta-base"
MAX_SEQ_LENGTH = 256
# Weak-label chunking: target chunk size in syllable tokens.
CHUNK_TOKENS = 150
