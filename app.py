# -*- coding: utf-8 -*-
"""
================================================================================
app.py — SMART SCANNER (Slanted-Margin Dewarp Vision + Layout Parser + KB Tags)
================================================================================
AI INSTRUCTION MANUAL (যেকোনো এআই বটের জন্য নির্দেশনা):
1. এই প্রোগ্রামের সমস্ত নিয়ম `knowledge_base.json` ফাইলে সংরক্ষিত আছে।
2. বাঁকা বা ভাঁজ হওয়া পাতার জন্য এতে কলাম ডিওয়ার্পিং (`dewarp_column`) এবং র‍্যানস্যাক
   ঢালু মার্জিন ফিটিং (`fit_slanted_margin`) যুক্ত করা হয়েছে।
3. মার্জিনের প্রথম টোকেনের প্রস্থ (প্রশ্ন নম্বর বনাম 'উত্তর:'/'ব্যাখ্যা:'), অপশন বৃত্ত এবং
   নলেজ-বেস ট্যাগ একসাথে মিলিয়ে প্রতিটি প্রশ্ন নিখুঁতভাবে কাটা হয়।
================================================================================
"""

import os
os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # tesseract shared CPU-তে আটকে যাওয়া ঠেকায়
import re
import time
import glob
import json
import random
import base64
import cv2
import fitz
import numpy as np
import pandas as pd
import streamlit as st

# নিরাপদভাবে pytesseract ইমপোর্ট করা (পিসিতে না থাকলেও যেন কখনোই ক্র্যাশ না করে)
try:
    import pytesseract
    if os.name == "nt":
        for _t_path in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        ):
            if os.path.exists(_t_path):
                pytesseract.pytesseract.tesseract_cmd = _t_path
                break
except Exception:
    pytesseract = None

try:
    import streamlit.components.v1 as components
except Exception:
    components = None

from learning_engine import (
    load_knowledge_base,
    commit_rule_to_kb,
    analyze_user_corrections,
    verify_admin_pin,
    find_all_tag_conflicts,
    resolve_tag_conflicts,
    sync_all_kb_files,
    push_file_to_github,
    push_binary_file_to_github,
    trigger_github_yolo_training,
    get_github_training_status,
    CATEGORY_LABELS_BN,
    KB_FILENAME
)

# ১. পেজ কনফিগারেশন
st.set_page_config(
    page_title="SMART SCANNER • VISION v5.0",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ২. বাহ্যিক style.css ফাইল লোড করা
def load_external_css(css_file="style.css"):
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_external_css()

# ==============================================================================
# ৩. কেন্দ্রীয় নলেজ-বেস (knowledge_base.json) থেকে সব নিয়ম ও ট্যাগ লোড করা
# ==============================================================================
KB = load_knowledge_base()
Q_TAGS = KB.get("question_tags", [])
SOL_TAGS = KB.get("solution_tags", [])
HEADER_NOISE_LIST = KB.get("header_noise_patterns", [])
BULLET_STOP_LIST = KB.get("bullet_stop_chars", ["➢", "➤", "➣", "•", "♦", "►"])
OPTION_UNITS_LIST = KB.get("option_units", ["mol", "L", "g", "kg", "mL", "ml", "cm", "mm", "m", "gm", "sec", "min", "s", "×", "x", "10^"])
DOTLESS_KW_LIST = KB.get("dotless_question_keywords", [])
VALID_CROP_KW_LIST = KB.get("valid_crop_keywords", [])
EQ_GUARD_KW_LIST = KB.get("equation_question_guard_words", [])
LAYOUT_CFG = KB.get("layout_thresholds", {})

MAX_Q_NUMBER = int(LAYOUT_CFG.get("max_question_number", 160))
MIN_START_GAP_PT = float(LAYOUT_CFG.get("min_start_gap_pt", 16.0))
MAX_Q_X_OFFSET = float(LAYOUT_CFG.get("max_q_x_offset", 52.0))
MAX_SOL_X_OFFSET = float(LAYOUT_CFG.get("max_sol_x_offset", 120.0))
DEFAULT_BLANK_GAP_PT = int(float(LAYOUT_CFG.get("max_internal_blank_gap_pt", 24.0)))

def parse_box_selection_string(sel_str, max_count):
    sel_str = sel_str.strip()
    if not sel_str:
        return set(range(max_count))
    selected = set()
    for part in sel_str.split(","):
        part = part.strip()
        if "-" in part:
            b = part.split("-", 1)
            if b[0].strip().isdigit() and b[1].strip().isdigit():
                for idx in range(max(1, int(b[0])), min(max_count, int(b[1])) + 1):
                    selected.add(idx - 1)
        elif part.isdigit():
            idx = int(part)
            if 1 <= idx <= max_count:
                selected.add(idx - 1)
    return selected

# ৪. মিনিমাল সাইডবার ও পিন-সুরক্ষিত লাইভ ট্রেনিং প্যানেল
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <span class="sb-brand-left">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#5E9E8B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
            <span>সেটিংস</span>
        </span>
        <span class="sb-engine-pill">VISION v5.0</span>
    </div>
    """, unsafe_allow_html=True)

    with st.popover("পেজ সেটআপ", use_container_width=True):
        PAGE_MODE = st.radio("মোড", ["অটো", "বই", "স্লাইড"], index=0, horizontal=True)
        COLUMNS_PER_PAGE = st.radio("কলাম", [1, 2], index=1, horizontal=True)
        PAGE_RANGE_TEXT = st.text_input("পৃষ্ঠা", "", placeholder="1-5 বা ফাঁকা")
        HEADER_TITLE = st.text_input("শিরোনাম", "MCQ Practice Sheet")
        INVERT_DARK_OUT = st.checkbox("সাদা ব্যাকগ্রাউন্ড", value=False)

    with st.expander("মার্জিন ও মাপ", expanded=False):
        QUESTION_GAP_PT = st.slider("গ্যাপ (pt)", 4, 40, 12, step=2)
        MIN_Q_HEIGHT_PT = st.slider("ন্যূনতম উচ্চতা", 10, 45, 16, step=2)
        RIGHT_EDGE_TRIM_PT = st.slider("ডান ছাঁটাই", 0, 35, 0, step=1)
        MAX_BLANK_GAP_PT = st.slider(
            "সর্বোচ্চ ভেতরের গ্যাপ",
            12, 80, DEFAULT_BLANK_GAP_PT, step=2,
            help="প্রশ্নের নিচে উত্তর লেখার জন্য বড় ফাঁকা জায়গা থাকলে এই মাপের পর বক্সটি স্বয়ংক্রিয়ভাবে কেটে যাবে।"
        )

    with st.expander("ব্যাচ এডিট", expanded=False):
        BATCH_TARGET = st.selectbox(
            "টার্গেট",
            ["সব বক্স", "নির্দিষ্ট বক্স", "বাম কলাম", "ডান কলাম", "নির্দিষ্ট পেজ"]
        )
        BATCH_BOX_NUMS = ""
        BATCH_PAGE_NO = 1
        if BATCH_TARGET == "নির্দিষ্ট বক্স":
            BATCH_BOX_NUMS = st.text_input("বক্স নং", "1-5", placeholder="1-4, 7")
        elif BATCH_TARGET == "নির্দিষ্ট পেজ":
            BATCH_PAGE_NO = st.number_input("পেজ নং", min_value=1, value=1, step=1)

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            BATCH_DX = st.number_input("ডানে/বামে (X)", value=0.0, step=2.0, format="%.1f")
            BATCH_DW = st.number_input("প্রস্থ (W)", value=0.0, step=2.0, format="%.1f")
        with b_col2:
            BATCH_DY = st.number_input("নিচে/উপরে (Y)", value=0.0, step=2.0, format="%.1f")
            BATCH_DH = st.number_input("উচ্চতা (H)", value=0.0, step=2.0, format="%.1f")

        APPLY_BATCH_BTN = st.button("প্রয়োগ")

    with st.expander("স্মার্ট ট্রেনিং ও এক্স-রে", expanded=False):
        XRAY_MODE = st.checkbox("এক্স-রে মোড", value=False)

        if not st.session_state.get("is_admin_unlocked", False):
            pin_in = st.text_input("পিন", type="password", placeholder="🔒 পিন...", key="sb_pin_input", label_visibility="collapsed")
            if st.button("আনলক", key="btn_unlock_admin"):
                if verify_admin_pin(pin_in):
                    st.session_state["is_admin_unlocked"] = True
                    st.rerun()
                else:
                    st.error("ভুল পিন!")
        else:
            RULE_CAT_LABELS = {
                "সমাধান": "solution_tags",
                "প্রশ্ন": "question_tags",
                "জলছাপ": "header_noise_patterns",
                "বুলেট": "bullet_stop_chars",
                "কীওয়ার্ড": "valid_crop_keywords"
            }
            CAT_KEY_TO_LABEL = {v: k for k, v in RULE_CAT_LABELS.items()}

            sel_cat_label = st.selectbox("ক্যাটাগরি", list(RULE_CAT_LABELS.keys()))
            manual_rule_val = st.text_input("ট্যাগ", "", placeholder="ট্যাগ লিখুন...")

            if st.button("✓ সেভ", key="btn_manual_rule"):
                if manual_rule_val.strip():
                    cat_key = RULE_CAT_LABELS[sel_cat_label]
                    raw_in = manual_rule_val.strip()
                    ok, msg = commit_rule_to_kb(cat_key, raw_in, "Manual rule", force_move=False)
                    if ok:
                        st.session_state.pop("sb_conflict_state", None)
                        st.toast("সেভ হয়েছে!")
                        st.rerun()
                    elif msg.startswith("CONFLICT|"):
                        parts = msg.split("|", 2)
                        st.session_state["sb_conflict_state"] = {
                            "val": raw_in,
                            "target_cat": cat_key,
                            "other_cats": parts[1].split(",")
                        }
                    else:
                        st.warning(msg)

            c_state = st.session_state.get("sb_conflict_state")
            if c_state:
                other_short = ", ".join(CAT_KEY_TO_LABEL.get(c, c) for c in c_state["other_cats"])
                st.error(f"⚠️ '{c_state['val']}' আছে: {other_short}")
                choose_options = {
                    "প্রশ্ন": "question_tags",
                    "সমাধান": "solution_tags",
                    "জলছাপ": "header_noise_patterns"
                }
                def_label = CAT_KEY_TO_LABEL.get(c_state["target_cat"], "প্রশ্ন")
                opt_list = list(choose_options.keys())
                def_idx = opt_list.index(def_label) if def_label in opt_list else 0
                resolved_label = st.selectbox(
                    "রাখুন:",
                    opt_list,
                    index=def_idx,
                    key="sb_conflict_resolve_sel"
                )
                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("✓ স্থানান্তর", key="btn_confirm_move"):
                        chosen_k = choose_options[resolved_label]
                        commit_rule_to_kb(chosen_k, c_state["val"], f"Moved to {chosen_k}", force_move=True)
                        st.session_state.pop("sb_conflict_state", None)
                        st.toast("স্থানান্তরিত!")
                        st.rerun()
                with rc2:
                    if st.button("✕ বাতিল", key="btn_cancel_move"):
                        st.session_state.pop("sb_conflict_state", None)
                        st.rerun()

            st.markdown("---")
            conflicts_list, _ = find_all_tag_conflicts(KB)
            cat_choice_map = {
                "প্রশ্ন": "question_tags",
                "সমাধান": "solution_tags",
                "জলছাপ": "header_noise_patterns",
                "মুছুন": "DELETE"
            }
            rev_choice_map = {v: k for k, v in cat_choice_map.items()}

            if conflicts_list:
                st.warning(f"🧹 ডুপ্লিকেট: {len(conflicts_list)}টি")
                resolutions_map = {}
                choice_keys = list(cat_choice_map.keys())
                for c_idx, c_item in enumerate(conflicts_list):
                    def_lbl = rev_choice_map.get(c_item.get("default_cat"), "প্রশ্ন")
                    d_idx = choice_keys.index(def_lbl) if def_lbl in choice_keys else 0
                    sel_res = st.selectbox(
                        f"'{c_item['display']}' • [{c_item['where']}]",
                        choice_keys,
                        index=d_idx,
                        key=f"dup_fix_{c_idx}_{c_item['norm_key']}"
                    )
                    resolutions_map[c_item["norm_key"]] = cat_choice_map[sel_res]

                if st.button("🧹 ফিক্স", key="btn_fix_all_dups"):
                    resolve_tag_conflicts(resolutions_map)
                    st.toast("ডুপ্লিকেট ফিক্সড!")
                    st.rerun()
            else:
                st.caption("✅ ডুপ্লিকেট নেই")

            with st.popover("🔍 ট্যাগ তালিকা", use_container_width=True):
                all_tag_items = []
                for ck, clbl in [("question_tags", "প্রশ্ন"), ("solution_tags", "সমাধান"), ("header_noise_patterns", "জলছাপ")]:
                    for t_val in KB.get(ck, []):
                        disp_t = re.sub(r"\\([.\:\-\(\)\[\]\s])", r"\1", str(t_val).strip())
                        all_tag_items.append((f"{disp_t} • [{clbl}]", disp_t, ck))

                if all_tag_items:
                    sel_tag_str = st.selectbox("ট্যাগ", [x[0] for x in all_tag_items], key="inspect_tag_sel")
                    picked = next(x for x in all_tag_items if x[0] == sel_tag_str)
                    cur_lbl = rev_choice_map.get(picked[2], "প্রশ্ন")
                    c_keys = list(cat_choice_map.keys())
                    new_lbl = st.selectbox("রাখুন", c_keys, index=c_keys.index(cur_lbl), key="inspect_cat_sel")
                    if st.button("✓ আপডেট", key="btn_inspect_update"):
                        target_c = cat_choice_map[new_lbl]
                        resolve_tag_conflicts({picked[1].lower(): target_c}, f"Updated {picked[1]}")
                        st.toast("আপডেট হয়েছে!")
                        st.rerun()

            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("🔄 সিঙ্ক", key="btn_manual_sync"):
                    sync_all_kb_files(KB, "Manual sync")
                    st.toast("সিঙ্ক সম্পন্ন!")
                    st.rerun()
            with bc2:
                if st.button("🔒 লক", key="btn_lock_admin"):
                    st.session_state["is_admin_unlocked"] = False
                    st.rerun()

        st.markdown("---")
        st.caption("🤖 **১-ক্লিক ক্লাউড এআই মডেল ট্রেইনিং**")
        if st.button("🚀 মডেল ট্রেইন ও আপডেট (.pt)", key="btn_cloud_yolo_train", use_container_width=True):
            ok_tr, msg_tr = trigger_github_yolo_training()
            if ok_tr:
                st.success(msg_tr)
            else:
                st.error(msg_tr)

        if st.button("📊 ট্রেইনিং স্ট্যাটাস দেখুন", key="btn_check_yolo_status", use_container_width=True):
            st.info(get_github_training_status())
        st.download_button(
            label="⬇ ব্যাকআপ",
            data=json.dumps(KB, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=KB_FILENAME,
            mime="application/json",
            key="dl_kb_backup"
        )

    mode_short = "AUTO" if PAGE_MODE == "অটো" else ("BOOK" if PAGE_MODE == "বই" else "SLIDE")
    range_short = PAGE_RANGE_TEXT.strip() if PAGE_RANGE_TEXT.strip() else "ALL"

    st.markdown(f"""
    <div class="sb-summary-card">
        <div class="sb-summary-row"><span>মোড</span><span class="sb-summary-val">{mode_short} • {COLUMNS_PER_PAGE}C</span></div>
        <div class="sb-summary-row"><span>পৃষ্ঠা • গ্যাপ</span><span class="sb-summary-val">{range_short} • {QUESTION_GAP_PT}pt</span></div>
        <div class="sb-summary-row"><span>ট্যাগ</span><span class="sb-summary-val">{len(Q_TAGS)}Q • {len(SOL_TAGS)}S</span></div>
    </div>
    """, unsafe_allow_html=True)

    try:
        from info_panel import render_info_panel
        render_info_panel()
    except Exception:
        pass

# ==============================================================================
# ৫. ডায়নামিক রেগুলার এক্সপ্রেশন ইঞ্জিন (knowledge_base.json থেকে চালিত)
# ==============================================================================
SC = 2.5

q_prefix_pattern = "|".join(re.escape(t) for t in sorted(Q_TAGS, key=len, reverse=True)) if Q_TAGS else "Q|প্রশ্ন"
sol_prefix_pattern = "|".join(re.escape(t) for t in sorted(SOL_TAGS, key=len, reverse=True)) if SOL_TAGS else "Sol|সমাধান|উত্তর"
bullet_chars_escaped = "".join(re.escape(c) for c in BULLET_STOP_LIST)
unit_pattern = "|".join(re.escape(u) for u in sorted(OPTION_UNITS_LIST, key=len, reverse=True))
dotless_kw_pattern = "|".join(re.escape(w) for w in DOTLESS_KW_LIST) if DOTLESS_KW_LIST else "কত|কোনটি"
valid_crop_kw_pattern = "|".join(re.escape(w) for w in VALID_CROP_KW_LIST) if VALID_CROP_KW_LIST else "কত|কোনটি"
eq_guard_kw_pattern = "|".join(re.escape(w) for w in EQ_GUARD_KW_LIST) if EQ_GUARD_KW_LIST else "কত|কোনটি|হলে|যদি"
noise_pattern_str = "|".join(HEADER_NOISE_LIST) if HEADER_NOISE_LIST else "Educationalpdf"

NUM_RE = re.compile(
    rf"^[\(\[]?(?:(?:{q_prefix_pattern})\s*[\.\:\-\ঃ\/\|]*\s*)?([0-9০-৯]{{1,3}})(?:[\.\)\]\:\ঃ\…\-\?]+|(?=[A-Za-zঅ-ঔক-হ]))(?![0-9০-৯])",
    re.IGNORECASE
)
PURE_PREFIX_RE = re.compile(rf"^(?:{q_prefix_pattern})[\.\:\-\ঃ\/\|]*\s*[0-9০-৯]*$", re.IGNORECASE)
OPT_TOKEN_RE = re.compile(r"^(?:[\(\[]?[a-dA-Dক-ঘ][\)\]\.]|[ⓐⓑⓒⓓⓔ①②③④⑧⑨])$")
BULLET_STOP_RE = re.compile(rf"^[{bullet_chars_escaped}]\s*")
OPT_UNIT_RE = re.compile(rf"^(?:{unit_pattern}|10\^?)$", re.IGNORECASE)
BOARD_TAG_RE = re.compile(r"[\[\(].*(?:1[0-9]|2[0-6]).*[\]\)]")

SOL_WORD_CORE = (
    r"(?:[sS58\$][oO0aAeE][lL1Ii\|!t](?:[nNmMrRuU23\^\°\*\~\"\'\`\’\”\ⁿ\-\_]+)?|"
    r"solution|ans(?:wer)?|shortcut|hints?|note|"
    rf"{sol_prefix_pattern})"
)

SOL_TOKEN_RE = re.compile(
    rf"^[\(\[\{{•\*{bullet_chars_escaped}\-\—\–\.\:\;\,\'\"\`\~\|]*{SOL_WORD_CORE}(?:$|[\s\:\ঃ\.\;\,\-\—\–\(\[\{{\"'\`\’\”\ⁿ\^\°\*\/\|0-9০-৯])",
    re.IGNORECASE
)

INLINE_SOL_RE = re.compile(
    rf"(?:^|\s|[\]\)\।\|]){SOL_WORD_CORE}(?:\s*[\:\ঃ\-\—\–\.\(\[]|\s+[nNⁿ\^]\b)",
    re.IGNORECASE
)

HEADER_NOISE_RE = re.compile(rf"(?:{noise_pattern_str})", re.IGNORECASE)
DOTLESS_KW_RE = re.compile(rf"(?:{dotless_kw_pattern})\b", re.IGNORECASE)
VALID_CROP_KW_RE = re.compile(rf"(?:{valid_crop_kw_pattern})", re.IGNORECASE)
EQ_GUARD_KW_RE = re.compile(rf"(?:{eq_guard_kw_pattern})", re.IGNORECASE)

def _b2e_int(s):
    t = s.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")).strip()
    return int(t) if t.isdigit() else None

def is_solution_text(text):
    if not text:
        return False
    clean = re.sub(r"[\u200c\u200d\ufeff]", "", text).strip()
    if not clean:
        return False
    if SOL_TOKEN_RE.match(clean):
        return True
    stripped = clean.lstrip("([{•*-➢➤.@©\"'` ").rstrip(".:;ঃ-ⁿ^°*~\"'`)]} ")
    if not stripped:
        return False
    if SOL_TOKEN_RE.match(stripped):
        return True
    s_low = stripped.lower()
    head_part = re.split(r"[\:\ঃ\(\[\.\-]", s_low)[0].strip()
    for tag in SOL_TAGS:
        t_low = tag.lower().strip()
        if not t_low:
            continue
        if s_low == t_low or s_low.startswith(t_low + ":") or s_low.startswith(t_low + "ঃ") or (head_part and head_part == t_low):
            return True
        if len(t_low) >= 4 and s_low.startswith(t_low):
            return True
    return False

def line_has_solution_marker(line, joined_text, max_sol_x):
    if not line:
        return False, None
    wx0, wy0, _, _, t0 = line[0]

    if wx0 <= max_sol_x and is_solution_text(t0):
        return True, wy0

    if len(line) >= 2 and wx0 <= max_sol_x:
        t1 = line[1][4].strip()
        combined_01 = f"{t0}{t1}"
        if is_solution_text(combined_01) or (re.match(r"^[sS58\$][oO0aAeE][lL1Ii\|!t]$", t0.strip("([. ")) and re.match(r"^[nNmMrR23\^\°\*\ⁿ\:\.\-]+$", t1)):
            return True, wy0

    for idx_w, w in enumerate(line[:6]):
        if w[0] <= max_sol_x:
            if is_solution_text(w[4]):
                return True, w[1]
            if idx_w + 1 < len(line):
                comb = f"{w[4]}{line[idx_w + 1][4]}"
                if is_solution_text(comb):
                    return True, w[1]

    if INLINE_SOL_RE.search(joined_text):
        return True, wy0

    return False, None

# ==============================================================================
# ৬. ইমেজ প্রসেসিং, ডিওয়ার্পিং, ঢালু মার্জিন ও অপশন সার্কেল ডিটেকশন
# ==============================================================================
def parse_page_range(range_str, total_pages):
    range_str = range_str.strip()
    if not range_str:
        return list(range(total_pages))
    selected = set()
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            b = part.split("-", 1)
            if b[0].strip().isdigit() and b[1].strip().isdigit():
                for p in range(max(1, int(b[0])), min(total_pages, int(b[1])) + 1):
                    selected.add(p - 1)
        elif part.isdigit():
            p = int(part)
            if 1 <= p <= total_pages:
                selected.add(p - 1)
    return sorted(selected) if selected else list(range(total_pages))

def page_to_gray(page, scale=SC):
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY)
    g = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    if g.mean() < 110:
        g = 255 - g
    return g

def clip_is_dark(page, clip):
    pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5), clip=clip, colorspace=fitz.csGRAY, alpha=False)
    return float(np.frombuffer(pix.samples, np.uint8).mean()) < 110

def normalize_shadow_gray(gray_img):
    h, w = gray_img.shape[:2]
    small = cv2.resize(gray_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg_small = cv2.medianBlur(small, 31)
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.divide(gray_img, bg, scale=255)

def get_ink_mask(gray_img):
    norm = normalize_shadow_gray(gray_img)
    blur = cv2.GaussianBlur(norm, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 14)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return (clean > 0) | (norm < 188)

def count_option_circles(gray_img, x0_pt, y0_pt, x1_pt, y1_pt):
    """ওপেনসিভি দিয়ে অপশনের গোল বৃত্ত (ক, খ, গ, ঘ বা ⓐ, ⓑ, ⓒ, ⓓ) গণনা করে।"""
    x0, y0 = max(0, int(x0_pt * SC)), max(0, int(y0_pt * SC))
    x1, y1 = min(gray_img.shape[1], int(x1_pt * SC)), min(gray_img.shape[0], int(y1_pt * SC))
    if x1 - x0 < 18 or y1 - y0 < 7:
        return 0
    roi = gray_img[y0:y1, x0:x1]
    bw = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = 0
    min_d, max_d = int(4.8 * SC), int(17.5 * SC)
    for c in cnts:
        bx, by, bw_c, bh_c = cv2.boundingRect(c)
        if min_d <= bw_c <= max_d and min_d <= bh_c <= max_d:
            aspect = bw_c / float(max(1, bh_c))
            if 0.65 <= aspect <= 1.48:
                area = cv2.contourArea(c)
                if area > 0.28 * (bw_c * bh_c):
                    circles += 1
    return circles

def find_divider(gray_img, W, H):
    x_start, x_end = int(W * 0.34 * SC), int(W * 0.66 * SC)
    y_start, y_end = int(H * 0.08 * SC), int(H * 0.92 * SC)
    ink_full = get_ink_mask(gray_img).astype(np.uint8)
    roi = cv2.dilate(ink_full[y_start:y_end, x_start:x_end], np.ones((1, 3), np.uint8), iterations=1)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(30 * SC)))
    v_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, v_kernel)
    col_sum = v_lines.sum(axis=0)
    if col_sum.max() > int(30 * SC):
        return (x_start + int(np.argmax(col_sum))) / SC, True
    ink_sum = roi.sum(axis=0)
    row_cover = float((roi.sum(axis=1) > 0).mean())
    if ink_sum.min() == 0 and row_cover >= 0.28:
        zero = (ink_sum == 0).astype(np.int8)
        best_len, best_start, run_start = 0, 0, None
        for i, v in enumerate(list(zero) + [0]):
            if v and run_start is None:
                run_start = i
            elif not v and run_start is not None:
                if i - run_start > best_len:
                    best_len, best_start = i - run_start, run_start
                run_start = None
        if best_len >= int(7 * SC):
            return (x_start + best_start + best_len / 2.0) / SC, True
    return W / 2.0, False

def trace_slanted_divider(gray_img, W, H):
    """বাঁকা কলাম বিভাজক রেখার ঢাল (Slope) ও অবস্থান নির্ণয় করে।"""
    h_px, w_px = gray_img.shape[:2]
    ink_u8 = get_ink_mask(gray_img).astype(np.uint8)
    v_open = cv2.morphologyEx(ink_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, int(h_px * 0.04)))))
    lo, hi = int(w_px * 0.30), int(w_px * 0.70)
    yy, xx = np.nonzero(v_open[:, lo:hi])
    if len(xx) > int(h_px * 0.25):
        xx = xx + lo
        best_sc, best_s, best_x0 = 0, 0.0, w_px * 0.5
        for s in np.arange(-0.07, 0.0701, 0.004):
            x_proj = np.round(xx - s * yy).astype(int)
            mn = int(x_proj.min())
            hist = np.bincount(x_proj - mn)
            smooth = np.convolve(hist, np.ones(5), mode="same")
            idx = int(np.argmax(smooth))
            if smooth[idx] > best_sc:
                best_sc = float(smooth[idx])
                best_s = float(s)
                best_x0 = float(idx + mn)
        if best_sc >= h_px * 0.28:
            mid_div_pt = (best_x0 + best_s * (h_px * 0.5)) / SC
            return mid_div_pt, True, best_s, lambda y_pt: float((best_x0 / SC) + best_s * float(y_pt))

    base_div, is_two_col = find_divider(gray_img, W, H)
    return base_div, is_two_col, 0.0, lambda _y: base_div

def snap_cut_line(ink, xr, y_target, up_win=2.5, down_win=1.0):
    h_px, w_px = ink.shape
    x0, x1 = max(0, int(xr[0] * SC)), min(w_px, int((xr[1] - 8.0) * SC))
    a = max(0, int((y_target - up_win) * SC))
    b = min(h_px, int((y_target + down_win) * SC))
    if b <= a or x1 <= x0:
        return y_target
    prof = ink[a:b, x0:x1].mean(axis=1)
    best_row = int(np.argmin(prof))
    return (a + best_row) / SC

def trim_vertical_whitespace(ink, xr, top, bot, max_gap_pt=None):
    if max_gap_pt is None:
        max_gap_pt = float(MAX_BLANK_GAP_PT)

    h_px, w_px = ink.shape
    x0 = max(0, int((xr[0] + 5.0) * SC))
    x1 = min(w_px, int((xr[1] - 9.0) * SC))
    y0, y1 = max(0, int(top * SC)), min(h_px, int(bot * SC))
    if y1 <= y0 or x1 <= x0:
        return top, bot

    roi = ink[y0:y1, x0:x1].astype(np.uint8)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(14, int(18 * SC))))
    v_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, v_kernel)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, int((x1 - x0) * 0.48)), 1))
    h_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, h_kernel)
    clean_roi = roi & (~v_lines) & (~h_lines)

    row_ink = clean_roi.mean(axis=1) > 0.003
    nz = np.where(row_ink)[0]
    if len(nz) == 0:
        return top, bot

    first_row = nz[0]
    last_row = nz[0]
    max_gap_px = int(float(max_gap_pt) * SC)
    min_q_px = int(10.0 * SC)

    for r_idx in nz[1:]:
        if (r_idx - last_row) > max_gap_px and (last_row - first_row) >= min_q_px:
            break
        last_row = r_idx

    return max(top, (y0 + first_row) / SC - 3.5), min(bot, (y0 + last_row) / SC + 5.0)

# ==============================================================================
# ৭. ইউনিভার্সাল ভিশন + ট্রেইনড YOLO AI ফিউশন ইঞ্জিন (ACS + উদ্ভাস + যেকোনো পিডিএফ)
# ==============================================================================
def _best_local_tilt_angle(ink_seg, rng=5.5, step=0.25):
    if ink_seg.size == 0 or ink_seg.sum() < 150:
        return 0.0
    small = cv2.resize(ink_seg.astype(np.float32), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    center = (sw * 0.5, sh * 0.5)
    best_score, best_ang = -1.0, 0.0
    for ang in np.arange(-rng, rng + 1e-6, step):
        M = cv2.getRotationMatrix2D(center, float(ang), 1.0)
        rot = cv2.warpAffine(small, M, (sw, sh))
        proj = rot.sum(axis=1)
        sc = float(np.sum(np.diff(proj) ** 2))
        if sc > best_score:
            best_score, best_ang = sc, float(ang)
    return best_ang

def _dewarp_column_arrays(norm_col, clean_ink_col, raw_ink_col):
    H_c, W_c = clean_ink_col.shape[:2]
    win = max(120, int(150 * SC))
    step = max(60, int(75 * SC))
    ys = sorted(set(list(range(0, max(1, H_c - win), step)) + [max(0, H_c - win)]))
    centers, angles = [], []
    for y0 in ys:
        seg = clean_ink_col[y0:min(H_c, y0 + win), :]
        ang = _best_local_tilt_angle(seg)
        centers.append(y0 + win * 0.5)
        angles.append(np.deg2rad(ang))
    angles = np.array(angles, dtype=np.float32)
    if len(angles) >= 3:
        angles = np.array([np.median(angles[max(0, i - 1):i + 2]) for i in range(len(angles))], dtype=np.float32)
    thy = np.interp(np.arange(H_c, dtype=np.float32), centers, angles).astype(np.float32)
    xc = W_c * 0.5
    xx = np.tile(np.arange(W_c, dtype=np.float32), (H_c, 1))
    yy = np.tile(np.arange(H_c, dtype=np.float32)[:, None], (1, W_c))
    map_y = (yy + thy[:, None] * (xx - xc)).astype(np.float32)

    n_dew = cv2.remap(norm_col, xx, map_y, cv2.INTER_LINEAR, borderValue=255)
    c_dew = (cv2.remap(clean_ink_col, xx, map_y, cv2.INTER_NEAREST, borderValue=0) > 0).astype(np.uint8)
    r_dew = (cv2.remap(raw_ink_col, xx, map_y, cv2.INTER_NEAREST, borderValue=0) > 0).astype(np.uint8)
    return n_dew, c_dew, r_dew, thy

def _detect_header_bottom_px(raw_ink_col):
    H_c, W_c = raw_ink_col.shape[:2]
    joined = cv2.dilate(raw_ink_col, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(3.5 * SC)), 1)))
    min_w = max(30, int(W_c * 0.42))
    h_rule = cv2.morphologyEx(joined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_w, 1)))
    thick = cv2.morphologyEx(joined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_w, max(4, int(2.5 * SC)))))
    thin_rule = h_rule & (thick == 0).astype(np.uint8)
    rows = np.where(thin_rule[:int(H_c * 0.16)].sum(axis=1) > 0)[0]
    if len(rows) > 0:
        return int(rows.max()) + int(3.0 * SC)
    return int(36.0 * SC)

def _horizontal_line_bands(ink_col, y_start, y_end, min_h_px):
    sub = ink_col[y_start:y_end, :]
    prof = sub.sum(axis=1)
    thr = max(2, int(0.004 * ink_col.shape[1]))
    on = prof > thr
    raw_bands = []
    s = None
    for i, v in enumerate(on):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if (i - s) >= min_h_px:
                raw_bands.append((s + y_start, i + y_start))
            s = None
    if s is not None and (y_end - y_start - s) >= min_h_px:
        raw_bands.append((s + y_start, y_end))

    if not raw_bands:
        return []

    hs = sorted(b - a for a, b in raw_bands)
    med_h = max(int(8.0 * SC), hs[len(hs) // 2])
    final_bands = []
    for a, b in raw_bands:
        if (b - a) > 1.85 * med_h:
            seg = ink_col[a:b, :].sum(axis=1).astype(float)
            smooth = np.convolve(seg, np.ones(5) / 5.0, mode="same")
            mx = smooth.max()
            cuts = []
            step_w = max(6, int(med_h * 0.6))
            idx_c = step_w
            while idx_c < len(smooth) - step_w:
                win = smooth[max(0, idx_c - 2):idx_c + 3]
                if win.min() < 0.20 * mx and smooth[max(0, idx_c - step_w):idx_c].max() > 0.38 * mx and smooth[idx_c:idx_c + step_w].max() > 0.38 * mx:
                    cuts.append(idx_c)
                    idx_c += int(med_h * 0.75)
                else:
                    idx_c += 1
            pts = [0] + cuts + [len(smooth)]
            for j in range(len(pts) - 1):
                if (pts[j + 1] - pts[j]) >= min_h_px:
                    final_bands.append((a + pts[j], a + pts[j + 1]))
        else:
            final_bands.append((a, b))
    return final_bands

def _first_token_span(ink_col, y0, y1, gap_px):
    cols = ink_col[y0:y1, :].sum(axis=0) > 0
    xs = np.where(cols)[0]
    if len(xs) == 0:
        return None
    s = int(xs[0])
    e = s
    for x in xs[1:]:
        if int(x) - e > gap_px:
            break
        e = int(x)
    return s, e, int(xs[-1])

def _line_tokens_info(ink_col, y0, y1, gap_px):
    cols = ink_col[y0:y1, :].sum(axis=0) > 0
    xs = np.where(cols)[0]
    if len(xs) == 0:
        return None
    runs = []
    s = int(xs[0])
    e = s
    for x in xs[1:]:
        ix = int(x)
        if ix - e > gap_px:
            runs.append((s, e))
            s = ix
        e = ix
    runs.append((s, e))
    x0, ft_end = runs[0]
    tw = ft_end - x0 + 1
    sec_x0 = runs[1][0] if len(runs) >= 2 else None
    x1 = runs[-1][1]
    return {
        "y0": y0, "y1": y1,
        "x0": x0, "ft_end": ft_end, "tw": tw,
        "sec_x0": sec_x0, "x1": x1, "span": x1 - x0,
        "num_runs": len(runs)
    }

def _fit_slanted_margin_ransac(line_items):
    if len(line_items) < 4:
        b_def = float(min((it["x0"] for it in line_items), default=0.0))
        return 0.0, b_def, lambda _y, b=b_def: b

    pts = np.array([[0.5 * (it["y0"] + it["y1"]), float(it["x0"])] for it in line_items], dtype=float)
    tol = max(4.0, 3.8 * SC)
    rnd = random.Random(42)
    best = None
    min_dy = max(40.0, 55.0 * SC)

    for _ in range(450):
        i, j = rnd.sample(range(len(pts)), 2)
        if abs(pts[j, 0] - pts[i, 0]) < min_dy:
            continue
        a = (pts[j, 1] - pts[i, 1]) / (pts[j, 0] - pts[i, 0])
        if abs(a) > 0.09:
            continue
        b = pts[i, 1] - a * pts[i, 0]
        res = pts[:, 1] - (a * pts[:, 0] + b)
        inliers = int((np.abs(res) <= tol).sum())
        left_outliers = int((res < -tol).sum())
        score = inliers - 3 * left_outliers
        if best is None or score > best[0]:
            best = (score, float(a), float(b))

    if best is None:
        b_def = float(np.percentile(pts[:, 1], 12))
        return 0.0, b_def, lambda _y, b=b_def: b

    _, a, b = best
    res = pts[:, 1] - (a * pts[:, 0] + b)
    mask = np.abs(res) <= tol
    if int(mask.sum()) >= 3:
        a_ref, b_ref = np.polyfit(pts[mask, 0], pts[mask, 1], 1)
        a, b = float(a_ref), float(b_ref)
    return a, b, lambda y, a=a, b=b: float(a * float(y) + b)

def _ocr_digits_voting(strip_gray, y0, y1, x_start):
    if pytesseract is None:
        return None
    votes = {}
    widths_px = [int(w * SC) for w in (11, 13, 15, 17, 19, 22)]
    pad_y = max(2, int(1.5 * SC))
    pad_x = max(2, int(1.8 * SC))

    for w_px in widths_px:
        if votes and max(votes.values()) >= 150:
            break  # যথেষ্ট নিশ্চিত হলে বাকি ঘুরে আর সময় নষ্ট নয়
        crop = strip_gray[max(0, y0 - pad_y):min(strip_gray.shape[0], y1 + pad_y), max(0, x_start - pad_x):min(strip_gray.shape[1], x_start + w_px)]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
        up = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        bordered = cv2.copyMakeBorder(up, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        _, otsu = cv2.threshold(bordered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        eroded = cv2.erode(otsu, np.ones((2, 2), np.uint8))

        for variant in (otsu, bordered, eroded):
            for psm in (7, 8):
                try:
                    d = pytesseract.image_to_data(
                        variant,
                        config=f"--psm {psm} -c tessedit_char_whitelist=0123456789.",
                        output_type=pytesseract.Output.DICT,
                        timeout=6
                    )
                    for txt, cf in zip(d["text"], d["conf"]):
                        t = txt.strip()
                        c_val = int(cf) if str(cf).lstrip("-").isdigit() else -1
                        if not t or c_val < 25:
                            continue
                        m = re.fullmatch(r"(\d{1,3})(\.?)", t)
                        if m and (m.group(2) or len(m.group(1)) >= 2):
                            qn = int(m.group(1))
                            if 1 <= qn <= MAX_Q_NUMBER:
                                weight = c_val + (18 if m.group(2) else 0) + 10 * (len(m.group(1)) - 1)
                                votes[qn] = votes.get(qn, 0) + weight
                except Exception:
                    pass
    if not votes:
        return None
    best_q = max(votes, key=votes.get)
    return best_q, votes[best_q]

def group_words_into_lines(words):
    if not words:
        return []
    sorted_w = sorted(words, key=lambda w: ((w[1] + w[3]) * 0.5, w[0]))
    lines = []
    cur_line = [sorted_w[0]]
    cur_cy = (sorted_w[0][1] + sorted_w[0][3]) * 0.5
    for w in sorted_w[1:]:
        w_cy = (w[1] + w[3]) * 0.5
        if abs(w_cy - cur_cy) <= 6.4:
            cur_line.append(w)
            cur_cy = sum((x[1] + x[3]) * 0.5 for x in cur_line) / len(cur_line)
        else:
            cur_line.sort(key=lambda x: x[0])
            lines.append(cur_line)
            cur_line = [w]
            cur_cy = w_cy
    cur_line.sort(key=lambda x: x[0])
    lines.append(cur_line)
    return lines

# ==============================================================================
# ৭.১ উদ্ভাস ও ঝুলন্ত-মার্জিন ইঞ্জিন (টেবিল মাস্কিং + ঝুলন্ত প্রশ্ন নম্বর + ডান বৃত্ত)
# ==============================================================================
def _find_udvash_valid_zones(ink_s, div_px, h_px, w_px):
    top_lim = int(36.0 * SC)
    bot_lim = h_px - int(38.0 * SC)

    v_open = cv2.morphologyEx(ink_s, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(20.0 * SC))))
    center_strip = v_open[:, max(0, div_px - int(16.0 * SC)):min(w_px, div_px + int(16.0 * SC))]
    has_div = (center_strip.sum(axis=1) > 0).astype(np.uint8)
    bridged_div = cv2.dilate(has_div.reshape(-1, 1), np.ones((int(26.0 * SC), 1), np.uint8)).ravel() > 0

    j_h = cv2.dilate(ink_s, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(3.5 * SC)), 1)))
    wide_h = cv2.morphologyEx(j_h, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(w_px * 0.46), 1)))
    wide_rows = np.where(wide_h.sum(axis=1) > int(w_px * 0.42))[0]

    is_table_or_banner = np.zeros(h_px, dtype=bool)
    if len(wide_rows) > 0:
        distinct_y = []
        for y in wide_rows:
            if not distinct_y or (y - distinct_y[-1]) > int(4.0 * SC):
                distinct_y.append(int(y))
        for y in distinct_y:
            is_table_or_banner[max(0, y - int(4 * SC)):min(h_px, y + int(5 * SC))] = True
        for i in range(len(distinct_y) - 1):
            if (distinct_y[i + 1] - distinct_y[i]) <= int(95.0 * SC):
                is_table_or_banner[distinct_y[i]:distinct_y[i + 1] + 1] = True

    if int(bridged_div.sum()) < int(h_px * 0.12):
        active = ~is_table_or_banner
    else:
        active = bridged_div & (~is_table_or_banner)

    active[:top_lim] = False
    active[bot_lim:] = False

    zones = []
    s = None
    min_zh = int(40.0 * SC)
    for y_i, val in enumerate(active):
        if val and s is None:
            s = y_i
        elif not val and s is not None:
            if (y_i - s) >= min_zh:
                zones.append((max(top_lim, s - int(4.0 * SC)), min(bot_lim, y_i + int(4.0 * SC))))
            s = None
    if s is not None and (bot_lim - s) >= min_zh:
        zones.append((max(top_lim, s - int(4.0 * SC)), bot_lim))

    return zones

def _find_column_subbanners_y(r_dew, zy0, zy1):
    Hc, Wc = r_dew.shape[:2]
    joined = cv2.dilate(r_dew, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(3.0 * SC)), 1)))
    min_bw = int(Wc * 0.42)
    h_lines = cv2.morphologyEx(joined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_bw, 1)))
    ys = np.where(h_lines[zy0:zy1, :].sum(axis=1) > min_bw * 0.75)[0] + zy0
    if len(ys) == 0:
        return []
    distinct_y = []
    for y in ys:
        if not distinct_y or (y - distinct_y[-1]) > int(4.0 * SC):
            distinct_y.append(int(y))
    banners = []
    i = 0
    while i < len(distinct_y):
        paired = False
        for j in range(i + 1, len(distinct_y)):
            dy = distinct_y[j] - distinct_y[i]
            if int(11.0 * SC) <= dy <= int(34.0 * SC):
                banners.append((distinct_y[i], distinct_y[j]))
                i = j + 1
                paired = True
                break
        if not paired:
            i += 1
    return banners

def _find_udvash_right_answer_circle(n_dew, c_dew, y_top, y_bot):
    Hc, Wc = n_dew.shape[:2]
    rx0 = int(Wc * 0.82)
    ry0 = max(0, int(y_top))
    ry1 = min(Hc, int(y_bot))
    if Wc - rx0 < 12 or ry1 - ry0 < 10:
        return None

    roi = n_dew[ry0:ry1, rx0:Wc]
    bw = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_d, max_d = int(5.2 * SC), int(15.2 * SC)
    candidates = []
    for c in cnts:
        bx, by, bw_c, bh_c = cv2.boundingRect(c)
        if min_d <= bw_c <= max_d and min_d <= bh_c <= max_d:
            aspect = bw_c / float(max(1, bh_c))
            if 0.68 <= aspect <= 1.42:
                area = cv2.contourArea(c)
                if area >= 0.28 * (bw_c * bh_c):
                    abs_x0 = rx0 + bx
                    abs_y0 = ry0 + by
                    abs_y1 = abs_y0 + bh_c
                    left_gap = c_dew[abs_y0:abs_y1, max(0, abs_x0 - int(3.5 * SC)):max(1, abs_x0 - 1)]
                    if left_gap.size == 0 or float(left_gap.mean()) < 0.18:
                        candidates.append((abs_y0, abs_y1, abs_x0))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0]

def _try_extract_udvash_page(page, gray_img, ink_mask, W, H, pno, xray_logs=None):
    norm_gray = normalize_shadow_gray(gray_img)
    raw_ink = get_ink_mask(gray_img).astype(np.uint8)
    h_px, w_px = gray_img.shape[:2]

    div_x_pt, is_two_col, div_slope, div_fn = trace_slanted_divider(gray_img, W, H)
    if not is_two_col:
        div_x_pt = W * 0.5
        div_slope = 0.0
        div_fn = lambda _y: W * 0.5

    if abs(div_slope) > 0.002:
        M_shear = np.float32([[1.0, -div_slope, 0.0], [0.0, 1.0, 0.0]])
        norm_s = cv2.warpAffine(norm_gray, M_shear, (w_px, h_px), borderValue=255)
        ink_s = cv2.warpAffine(raw_ink, M_shear, (w_px, h_px), borderValue=0)
        div_px = int(round(div_fn(0.0) * SC))
    else:
        norm_s = norm_gray.copy()
        ink_s = raw_ink.copy()
        div_px = int(round(div_x_pt * SC))

    valid_zones = _find_udvash_valid_zones(ink_s, div_px, h_px, w_px)
    if not valid_zones:
        return None

    v_div = cv2.morphologyEx(ink_s, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(18, int(h_px * 0.08)))))
    clean_s = ink_s & (cv2.dilate(v_div, np.ones((1, 5), np.uint8)) == 0).astype(np.uint8)

    left_edge_px = int(12.0 * SC)
    right_edge_px = int(28.0 * SC)
    col_px_ranges = [
        (left_edge_px, max(left_edge_px + 50, div_px - 3)),
        (min(w_px - 50, div_px + 3), w_px - right_edge_px)
    ]

    min_band_h = max(5, int(3.3 * SC))
    tok_gap_px = max(3, int(2.8 * SC))

    hanging_q_count = 0
    right_circle_count = 0
    outer_wide_label_count = 0
    udvash_rects = []

    for (zy0, zy1) in valid_zones:
        for ci, (cx0_px, cx1_px) in enumerate(col_px_ranges):
            n_col = norm_s[:, cx0_px:cx1_px].copy()
            c_col = clean_s[:, cx0_px:cx1_px].copy()
            r_col = ink_s[:, cx0_px:cx1_px].copy()

            n_dew, c_dew, r_dew, thy = _dewarp_column_arrays(n_col, c_col, r_col)
            Hc, Wc = c_dew.shape[:2]

            eff_y0, eff_y1 = zy0, zy1
            sub_banners = _find_column_subbanners_y(r_dew, zy0, zy1)
            for (by0, by1) in sub_banners:
                if (by0 - zy0) <= int(42.0 * SC):
                    eff_y0 = max(eff_y0, by1 + int(4.0 * SC))
                else:
                    eff_y1 = min(eff_y1, by0 - int(4.0 * SC))
                    break

            if (eff_y1 - eff_y0) < int(32.0 * SC):
                continue

            c_dew[:eff_y0, :] = 0
            c_dew[eff_y1:, :] = 0

            lines = []
            for y0, y1 in _horizontal_line_bands(c_dew, eff_y0, eff_y1, min_band_h):
                info = _line_tokens_info(c_dew, y0, y1, tok_gap_px)
                if info is not None and info["span"] >= int(16.0 * SC):
                    lines.append(info)

            if len(lines) < 3:
                continue

            body_x_samples = []
            for l in lines:
                if l["span"] < int(24.0 * SC):
                    continue
                if (
                    int(4.2 * SC) <= l["tw"] <= int(18.0 * SC)
                    and l["sec_x0"] is not None
                    and int(10.5 * SC) <= (l["sec_x0"] - l["x0"]) <= int(28.5 * SC)
                ):
                    body_x_samples.append(float(l["sec_x0"]))
                elif l["x0"] < int(Wc * 0.35):
                    body_x_samples.append(float(l["x0"]))

            if not body_x_samples:
                continue

            bx_arr = np.array(body_x_samples, dtype=float)
            bins = np.arange(bx_arr.min() - 4.0, bx_arr.max() + 8.0, 3.0 * SC)
            if len(bins) >= 2:
                hist, edges = np.histogram(bx_arr, bins=bins)
                pk = int(np.argmax(hist))
                mode_val = 0.5 * (edges[pk] + edges[pk + 1])
                inliers = [v for v in bx_arr if abs(v - mode_val) <= 5.5 * SC]
                x_body = float(np.median(inliers)) if inliers else float(np.median(bx_arr))
            else:
                x_body = float(np.median(bx_arr))

            q_starts = []
            for l in lines:
                hang_dist = x_body - float(l["x0"])
                if int(10.5 * SC) <= hang_dist <= int(28.5 * SC):
                    if int(4.2 * SC) <= l["tw"] <= int(18.5 * SC) and l["span"] >= int(32.0 * SC):
                        if l["sec_x0"] is not None and abs(float(l["sec_x0"]) - x_body) <= int(8.5 * SC):
                            if not q_starts or (l["y0"] - q_starts[-1]["y0"]) >= int(MIN_START_GAP_PT * SC):
                                q_starts.append(l)
                    elif l["tw"] >= int(21.0 * SC):
                        outer_wide_label_count += 1

            hanging_q_count += len(q_starts)

            for qi, qs in enumerate(q_starts):
                next_y_lim = q_starts[qi + 1]["y0"] - int(2.0 * SC) if (qi + 1) < len(q_starts) else (eff_y1 - int(2.0 * SC))
                sub_lines = [l for l in lines if qs["y0"] <= l["y0"] < next_y_lim]
                if not sub_lines:
                    continue

                ans_circle = _find_udvash_right_answer_circle(
                    n_dew, c_dew, qs["y0"] + int(8.0 * SC), next_y_lim
                )

                if ans_circle is not None:
                    right_circle_count += 1
                    c_y0, c_y1, _c_x0 = ans_circle
                    bot_px = c_y1 + int(2.5 * SC)
                    for idx_l, l in enumerate(sub_lines):
                        if l["y0"] <= c_y1 + int(2.0 * SC) and l["y1"] >= c_y0 - int(2.0 * SC):
                            bot_px = max(bot_px, l["y1"] + int(2.5 * SC))
                            if idx_l + 1 < len(sub_lines):
                                nxt_l = sub_lines[idx_l + 1]
                                if (
                                    (nxt_l["y0"] - l["y1"]) <= int(11.0 * SC)
                                    and abs(float(nxt_l["x0"]) - x_body) <= int(7.5 * SC)
                                    and int(6.5 * SC) <= nxt_l["tw"] <= int(14.0 * SC)
                                ):
                                    bot_px = max(bot_px, nxt_l["y1"] + int(2.5 * SC))
                            break
                else:
                    bot_px = qs["y1"] + int(2.5 * SC)
                    seen_opts = False
                    for idx_l, l in enumerate(sub_lines[1:], 1):
                        near_body = abs(float(l["x0"]) - x_body) <= int(8.0 * SC)
                        is_opt_line = near_body and (int(6.5 * SC) <= l["tw"] <= int(14.8 * SC))
                        is_sol_line = near_body and (l["tw"] >= int(15.5 * SC))
                        if not seen_opts:
                            if is_opt_line:
                                seen_opts = True
                                bot_px = l["y1"] + int(2.5 * SC)
                            elif idx_l <= 4 and (l["y0"] - bot_px) <= int(16.0 * SC):
                                bot_px = l["y1"] + int(2.5 * SC)
                            else:
                                break
                        else:
                            if is_sol_line or not is_opt_line:
                                break
                            bot_px = l["y1"] + int(2.5 * SC)

                top_px = max(0, qs["y0"] - int(2.5 * SC))
                bot_px = min(Hc - 1, bot_px)
                if (bot_px - top_px) < int(MIN_Q_HEIGHT_PT * SC):
                    continue

                sub_ink = c_dew[top_px:bot_px, :]
                ys_nz, xs_nz = np.where(sub_ink > 0)
                if len(ys_nz) < 25:
                    continue

                bx0_col = max(0, int(xs_nz.min()) - int(3.5 * SC))
                bx1_col = min(Wc - 1, int(xs_nz.max()) + int(4.5 * SC))
                top_col = max(0, top_px + int(ys_nz.min()) - int(2.5 * SC))
                bot_col = min(Hc - 1, top_px + int(ys_nz.max()) + int(3.0 * SC))

                yc_mid = int(0.5 * (top_col + bot_col))
                shear_shift_x = div_slope * yc_mid if is_two_col else 0.0
                orig_x0_px = max(4.0, cx0_px + bx0_col + shear_shift_x)
                orig_x1_px = min(float(w_px - 4), cx0_px + bx1_col + shear_shift_x)

                q_x0_pt = orig_x0_px / SC
                q_y0_pt = top_col / SC
                q_x1_pt = orig_x1_px / SC - RIGHT_EDGE_TRIM_PT
                q_y1_pt = bot_col / SC

                if (q_x1_pt - q_x0_pt) >= 65.0 and (q_y1_pt - q_y0_pt) >= float(MIN_Q_HEIGHT_PT):
                    q_y0_pt, q_y1_pt = trim_vertical_whitespace(ink_mask, (q_x0_pt, q_x1_pt), q_y0_pt, q_y1_pt, MAX_BLANK_GAP_PT)
                    rect = fitz.Rect(q_x0_pt, q_y0_pt, q_x1_pt, q_y1_pt)
                    if not rect.is_empty and not rect.is_infinite:
                        udvash_rects.append(rect)

    if hanging_q_count >= 3 and (right_circle_count >= 2 or outer_wide_label_count == 0) and len(udvash_rects) >= 3:
        if xray_logs is not None:
            for idx_r, r in enumerate(udvash_rects, 1):
                xray_logs.append({
                    "পৃষ্ঠা": pno + 1, "কলাম": 1 if (r.x0 + r.x1) * 0.5 < W * 0.5 else 2,
                    "Y": round(r.y0, 1),
                    "সিদ্ধান্ত": f"উদ্ভাস ভিশন (#{idx_r})",
                    "লাইন": f"Box: ({round(r.x0,1)}, {round(r.y0,1)}) -> ({round(r.x1,1)}, {round(r.y1,1)})"
                })
        return udvash_rects

    return None

# ==============================================================================
# ৭.২ আপনার ১০০% সফল ও অক্ষত ACS ইঞ্জিন (sample22.pdf ও sample 2.pdf)
# ==============================================================================
def _extract_acs_slanted_page(page, gray_img, ink_mask, W, H, pno, xray_logs=None):
    norm_gray = normalize_shadow_gray(gray_img)
    raw_ink = get_ink_mask(gray_img).astype(np.uint8)
    h_px, w_px = gray_img.shape[:2]

    div_x_pt, is_two_col, div_slope, div_fn = trace_slanted_divider(gray_img, W, H)

    if is_two_col and abs(div_slope) > 0.002:
        M_shear = np.float32([[1.0, -div_slope, 0.0], [0.0, 1.0, 0.0]])
        norm_s = cv2.warpAffine(norm_gray, M_shear, (w_px, h_px), borderValue=255)
        ink_s = cv2.warpAffine(raw_ink, M_shear, (w_px, h_px), borderValue=0)
        div_px = int(round(div_fn(0.0) * SC))
    else:
        norm_s = norm_gray.copy()
        ink_s = raw_ink.copy()
        div_px = int(round(div_x_pt * SC))

    v_div = cv2.morphologyEx(ink_s, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(h_px * 0.10)))))
    clean_s = ink_s & (cv2.dilate(v_div, np.ones((1, 5), np.uint8)) == 0).astype(np.uint8)
    j_h = cv2.dilate(ink_s, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(3.0 * SC)), 1)))
    hr = cv2.morphologyEx(j_h, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(w_px * 0.16), 1)))
    thick_hr = cv2.morphologyEx(j_h, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(w_px * 0.16), max(4, int(2.5 * SC)))))
    hr = hr & (thick_hr == 0).astype(np.uint8)
    clean_s = clean_s & (cv2.dilate(hr, np.ones((3, 1), np.uint8)) == 0).astype(np.uint8)

    margin_edge_px = int(12.0 * SC)
    if is_two_col:
        col_px_ranges = [(margin_edge_px, max(margin_edge_px + 50, div_px - 3)), (min(w_px - 50, div_px + 3), w_px - margin_edge_px)]
    else:
        col_px_ranges = [(margin_edge_px, w_px - margin_edge_px)]

    pdf_words = [(w[0], w[1], w[2], w[3], w[4].strip()) for w in page.get_text("words") if w[4].strip()]
    pdf_lines_all = group_words_into_lines(pdf_words)

    col_data_list = []
    all_markers = []
    min_band_h = max(5, int(3.4 * SC))
    tok_gap_px = max(4, int(3.6 * SC))

    for ci, (cx0_px, cx1_px) in enumerate(col_px_ranges):
        n_col = norm_s[:, cx0_px:cx1_px].copy()
        c_col = clean_s[:, cx0_px:cx1_px].copy()
        r_col = ink_s[:, cx0_px:cx1_px].copy()

        if (not is_two_col or ci == 1) and c_col.shape[1] > int(20 * SC):
            c_col[:, -int(10 * SC):] = 0

        n_dew, c_dew, r_dew, thy = _dewarp_column_arrays(n_col, c_col, r_col)
        hdr_px = _detect_header_bottom_px(r_dew)
        c_dew[:hdr_px, :] = 0

        raw_lines = []
        for y0, y1 in _horizontal_line_bands(c_dew, hdr_px, h_px - int(24 * SC), min_band_h):
            sp = _first_token_span(c_dew, y0, y1, tok_gap_px)
            if sp is not None:
                raw_lines.append({"y0": y0, "y1": y1, "x0": sp[0], "ft_end": sp[1], "x1": sp[2]})

        if not raw_lines:
            continue

        _, _, mx_fn = _fit_slanted_margin_ransac(raw_lines)

        x_lo, x_hi = -int(3.2 * SC), int(32.0 * SC)
        strip_w = x_hi - x_lo
        Si = np.zeros((h_px, strip_w), dtype=np.uint8)
        Sg = np.full((h_px, strip_w), 255, dtype=np.uint8)
        Hc, Wc = c_dew.shape[:2]
        for y_r in range(Hc):
            a_x = int(round(mx_fn(y_r))) + x_lo
            s_x = max(0, a_x)
            e_x = min(Wc, a_x + strip_w)
            if e_x > s_x:
                Si[y_r, s_x - a_x:e_x - a_x] = c_dew[y_r, s_x:e_x]
                Sg[y_r, s_x - a_x:e_x - a_x] = n_dew[y_r, s_x:e_x]
        Si[:hdr_px, :] = 0

        col_markers = []
        max_lead_px = int(8.5 * SC)
        max_num_tw_px = int(19.5 * SC)

        for y0, y1 in _horizontal_line_bands(Si, hdr_px, h_px - int(22 * SC), min_band_h):
            xs = np.where(Si[y0:y1, :].sum(axis=0) > 0)[0]
            if len(xs) == 0 or int(xs[0]) > max_lead_px:
                continue

            sp = _first_token_span(Si, y0, y1, tok_gap_px)
            if sp is None:
                continue
            tw = sp[1] - sp[0] + 1

            full_xs = np.where(c_dew[y0:y1, :].sum(axis=0) > 0)[0]
            line_span_px = (int(full_xs[-1]) - int(full_xs[0])) if len(full_xs) > 0 else tw

            y0_pt = y0 / SC
            cx0_pt, cx1_pt = cx0_px / SC, cx1_px / SC
            matching_pdf_ln = [
                ln for ln in pdf_lines_all
                if ln and cx0_pt - 4 <= ln[0][0] <= cx1_pt + 4 and abs(ln[0][1] - y0_pt) <= 8.0
            ]
            pdf_line_txt = " ".join(w[4] for ln in matching_pdf_ln for w in ln).strip()

            if pdf_line_txt and HEADER_NOISE_RE.search(pdf_line_txt):
                continue

            is_sol_by_tag = False
            if matching_pdf_ln:
                is_sol_by_tag, _ = line_has_solution_marker(matching_pdf_ln[0], pdf_line_txt, cx0_pt + MAX_SOL_X_OFFSET)

            if is_sol_by_tag:
                col_markers.append({
                    "col": ci, "y0": y0, "y1": y1, "tw": tw,
                    "kind": "label", "num": None, "conf": 100, "guessed": False
                })
                continue

            read_res = _ocr_digits_voting(Sg, y0, y1, sp[0]) if tw <= int(24.0 * SC) else None

            if read_res is None and matching_pdf_ln:
                t0_pdf = matching_pdf_ln[0][0][4].strip("([{|! ")
                m_p = NUM_RE.match(t0_pdf)
                if m_p:
                    qn_p = _b2e_int(m_p.group(1))
                    if qn_p is not None and 1 <= qn_p <= MAX_Q_NUMBER:
                        read_res = (qn_p, 120)

            if read_res is not None:
                col_markers.append({
                    "col": ci, "y0": y0, "y1": y1, "tw": tw,
                    "kind": "num", "num": read_res[0], "conf": read_res[1], "guessed": False
                })
            elif tw <= max_num_tw_px and line_span_px >= int(42.0 * SC):
                col_markers.append({
                    "col": ci, "y0": y0, "y1": y1, "tw": tw,
                    "kind": "cand", "num": None, "conf": 0, "guessed": False
                })
            else:
                col_markers.append({
                    "col": ci, "y0": y0, "y1": y1, "tw": tw,
                    "kind": "label", "num": None, "conf": 0, "guessed": False
                })

        col_data_list.append({
            "ci": ci, "cx0_px": cx0_px, "cx1_px": cx1_px,
            "n_dew": n_dew, "c_dew": c_dew, "thy": thy,
            "mx_fn": mx_fn, "markers": col_markers
        })
        all_markers.extend(col_markers)

    if not all_markers:
        return []

    all_markers.sort(key=lambda m: (m["col"], m["y0"]))
    numbered = [m for m in all_markers if m["kind"] == "num" and m["num"] is not None]

    if numbered:
        n_len = len(numbered)
        dp = [(1, -1)] * n_len
        for i in range(n_len):
            for j in range(i):
                diff = numbered[i]["num"] - numbered[j]["num"]
                if 1 <= diff <= 5 and (dp[j][0] + 1) > dp[i][0]:
                    dp[i] = (dp[j][0] + 1, j)
        best_end = max(range(n_len), key=lambda k: dp[k][0])
        chain_ids = set()
        cur_k = best_end
        while cur_k != -1:
            chain_ids.add(id(numbered[cur_k]))
            cur_k = dp[cur_k][1]

        for m in numbered:
            if id(m) not in chain_ids:
                m["kind"] = "cand" if m["tw"] <= int(19.5 * SC) else "label"
                m["num"] = None

        kept = [m for m in all_markers if m["kind"] == "num"]

        for a_m, b_m in zip(kept, kept[1:]):
            miss = list(range(a_m["num"] + 1, b_m["num"]))
            if not miss:
                continue
            pool = [
                m for m in all_markers
                if m["kind"] == "cand" and (a_m["col"], a_m["y0"]) < (m["col"], m["y0"]) < (b_m["col"], b_m["y0"])
            ]
            for k_num, m_cand in zip(miss, pool[:len(miss)]):
                m_cand["kind"] = "num"
                m_cand["num"] = k_num
                m_cand["guessed"] = True

        first_m = kept[0]
        lead_cands = [
            m for m in all_markers
            if m["kind"] == "cand"
            and (m["col"], m["y0"]) < (first_m["col"], first_m["y0"])
            and any(x["kind"] == "label" and (m["col"], m["y0"]) < (x["col"], x["y0"]) < (first_m["col"], first_m["y0"]) for x in all_markers)
        ]
        for k_num, m_cand in zip(range(first_m["num"] - 1, 0, -1), reversed(lead_cands)):
            m_cand["kind"] = "num"
            m_cand["num"] = k_num
            m_cand["guessed"] = True

        kept = [m for m in all_markers if m["kind"] == "num"]
        last_m = kept[-1]
        tail_cands = [
            m for m in all_markers
            if m["kind"] == "cand"
            and (m["col"], m["y0"]) > (last_m["col"], last_m["y0"])
            and any(x["kind"] == "label" and (x["col"], x["y0"]) > (m["col"], m["y0"]) and x["col"] == m["col"] for x in all_markers)
        ]
        for k_num, m_cand in zip(range(last_m["num"] + 1, MAX_Q_NUMBER + 1), tail_cands):
            m_cand["kind"] = "num"
            m_cand["num"] = k_num
            m_cand["guessed"] = True
    else:
        for idx_m, m in enumerate(all_markers):
            if m["kind"] == "cand":
                nxt_in_col = [x for x in all_markers[idx_m + 1:] if x["col"] == m["col"]]
                if nxt_in_col and nxt_in_col[0]["kind"] == "label":
                    m["kind"] = "num"
                    m["guessed"] = True

    page_rects = []
    for c_info in col_data_list:
        ci = c_info["ci"]
        cx0_px = c_info["cx0_px"]
        c_dew = c_info["c_dew"]
        thy = c_info["thy"]
        Hc, Wc = c_dew.shape[:2]

        c_markers = sorted([m for m in all_markers if m["col"] == ci], key=lambda x: x["y0"])
        q_markers = [m for m in c_markers if m["kind"] == "num"]

        for qm in q_markers:
            after_m = [m for m in c_markers if m["y0"] > qm["y0"] + int(4.0 * SC)]
            stop_y_px = after_m[0]["y0"] - int(2.0 * SC) if after_m else (Hc - int(15.0 * SC))

            if after_m and after_m[0]["kind"] == "label":
                lb = after_m[0]
                right_zone_x0 = int(Wc * 0.55)
                lim_y = min(Hc - 1, lb["y1"] + int(5.0 * SC))
                if lim_y > stop_y_px:
                    row_ink_right = c_dew[stop_y_px:lim_y, right_zone_x0:Wc].sum(axis=1) > 0
                    if int(row_ink_right.sum()) > 2:
                        last_r = 0
                        blank_run = 0
                        for r_i, has_v in enumerate(row_ink_right):
                            if has_v:
                                last_r = r_i
                                blank_run = 0
                            else:
                                blank_run += 1
                                if blank_run >= 4 and r_i > (lb["y1"] - stop_y_px):
                                    break
                        stop_y_px = min(Hc - 1, stop_y_px + last_r + int(2.0 * SC))

            top_px = max(0, qm["y0"] - int(3.0 * SC))
            bot_px = min(Hc - 1, stop_y_px)
            if bot_px - top_px < int(MIN_Q_HEIGHT_PT * SC):
                continue

            sub_ink = c_dew[top_px:bot_px, :]
            ys_nz, xs_nz = np.where(sub_ink > 0)
            if len(ys_nz) < 25:
                continue

            bx0_col = max(0, int(xs_nz.min()) - int(4.0 * SC))
            bx1_col = min(Wc - 1, int(xs_nz.max()) + int(5.0 * SC))
            top_col = max(0, top_px + int(ys_nz.min()) - int(3.0 * SC))
            bot_col = min(Hc - 1, top_px + int(ys_nz.max()) + int(3.5 * SC))

            yc_mid = int(0.5 * (top_col + bot_col))
            ang_mid = float(thy[min(Hc - 1, max(0, yc_mid))])
            dy_pad = abs(ang_mid) * (Wc * 0.35)

            orig_y0_px = max(0.0, top_col - dy_pad)
            orig_y1_px = min(float(h_px - 2), bot_col + dy_pad)
            shear_shift_x = div_slope * yc_mid if is_two_col else 0.0
            orig_x0_px = max(4.0, cx0_px + bx0_col + shear_shift_x)
            orig_x1_px = min(float(w_px - 4), cx0_px + bx1_col + shear_shift_x)

            q_x0_pt = orig_x0_px / SC
            q_y0_pt = orig_y0_px / SC
            q_x1_pt = orig_x1_px / SC - RIGHT_EDGE_TRIM_PT
            q_y1_pt = orig_y1_px / SC

            if (q_x1_pt - q_x0_pt) >= 65.0 and (q_y1_pt - q_y0_pt) >= float(MIN_Q_HEIGHT_PT):
                q_y0_pt, q_y1_pt = trim_vertical_whitespace(ink_mask, (q_x0_pt, q_x1_pt), q_y0_pt, q_y1_pt, MAX_BLANK_GAP_PT)
                rect = fitz.Rect(q_x0_pt, q_y0_pt, q_x1_pt, q_y1_pt)
                if not rect.is_empty and not rect.is_infinite:
                    page_rects.append(rect)
                    if xray_logs is not None:
                        xray_logs.append({
                            "পৃষ্ঠা": pno + 1, "কলাম": ci + 1,
                            "Y": round(q_y0_pt, 1),
                            "সিদ্ধান্ত": f"ACS ভিশন (#{qm['num'] if qm['num'] else '?'})",
                            "লাইন": f"Box: ({round(q_x0_pt,1)}, {round(q_y0_pt,1)}) -> ({round(q_x1_pt,1)}, {round(q_y1_pt,1)})"
                        })

    return page_rects

def extract_page_questions_vision_hybrid(page, gray_img, ink_mask, W, H, pno, xray_logs=None):
    udvash_res = _try_extract_udvash_page(page, gray_img, ink_mask, W, H, pno, xray_logs=xray_logs)
    if udvash_res is not None:
        return udvash_res
    return _extract_acs_slanted_page(page, gray_img, ink_mask, W, H, pno, xray_logs=xray_logs)

# ==============================================================================
# ৮. স্লাইড মোড: বক্সে ঘেরা প্রশ্ন শনাক্ত, Sol^n ও ফাঁকা জায়গা ছাঁটাই
# ==============================================================================
def is_slide_page(W, H):
    if PAGE_MODE == "বই":
        return False
    if PAGE_MODE == "স্লাইড":
        return True
    return W > 1.2 * H

def _order_rects_reading(rects):
    rects = sorted(rects, key=lambda r: (r.y0, r.x0))
    rows, out = [], []
    for r in rects:
        if rows and abs(r.y0 - rows[-1][0].y0) < 30:
            rows[-1].append(r)
        else:
            rows.append([r])
    for row in rows:
        out.extend(sorted(row, key=lambda r: r.x0))
    return out

def box_has_question_number(page, gray, rect):
    try:
        words = [w for w in page.get_text("words")
                 if rect.x0 - 2 <= w[0] <= rect.x1 and rect.y0 - 2 <= w[1] <= rect.y1 and w[4].strip()]
        if words:
            top_y = min(w[1] for w in words)
            first_line = sorted([w for w in words if w[1] <= top_y + 12], key=lambda w: w[0])
            return any(NUM_RE.match(w[4].strip()) or PURE_PREFIX_RE.match(w[4].strip()) for w in first_line[:2])
        if pytesseract is not None:
            x0, y0 = max(0, int(rect.x0 * SC)), max(0, int(rect.y0 * SC))
            x1, y1 = int(rect.x1 * SC), int(min(rect.y1, rect.y0 + 45) * SC)
            txt = pytesseract.image_to_string(gray[y0:y1, x0:x1], lang="eng", config="--psm 7").strip()
            return bool(re.match(r"^\W{0,3}[\(\[]?\s*[0-9]{1,3}\s*[\)\]\.]", txt))
    except Exception:
        pass
    return False

def trim_solution_inside_slide_box(page, gray, ink, rect):
    cur_rect = rect
    try:
        if pytesseract is not None:
            x0_px, y0_px = max(0, int(rect.x0 * SC)), max(0, int(rect.y0 * SC))
            x1_px, y1_px = min(gray.shape[1], int(rect.x1 * SC)), min(gray.shape[0], int(rect.y1 * SC))
            if y1_px - y0_px >= 40 and x1_px - x0_px >= 60:
                roi = gray[y0_px:y1_px, x0_px:x1_px]
                bw = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 13)
                data = pytesseract.image_to_data(bw, lang="eng+ben", config="--psm 6", output_type=pytesseract.Output.DICT)
                words = []
                for i in range(len(data["text"])):
                    t = data["text"][i].strip()
                    if not t:
                        continue
                    wx0 = rect.x0 + data["left"][i] / SC
                    wy0 = rect.y0 + data["top"][i] / SC
                    wx1 = wx0 + data["width"][i] / SC
                    wy1 = wy0 + data["height"][i] / SC
                    words.append((wx0, wy0, wx1, wy1, t))
                lines = group_words_into_lines(words)
                for ln in lines:
                    joined = " ".join(w[4] for w in ln)
                    has_s, sy = line_has_solution_marker(ln, joined, rect.x0 + (rect.width * 0.55))
                    if has_s and sy is not None and (sy - rect.y0) >= 18.0:
                        new_bot = snap_cut_line(ink, (rect.x0 + 4, rect.x1 - 4), sy - 1.5, up_win=2.5, down_win=0.5)
                        cur_rect = fitz.Rect(rect.x0, rect.y0, rect.x1, max(rect.y0 + 18.0, new_bot))
                        break
    except Exception:
        pass

    try:
        _, trimmed_bot = trim_vertical_whitespace(
            ink,
            (cur_rect.x0 + 6.0, cur_rect.x1 - 6.0),
            cur_rect.y0 + 4.0,
            cur_rect.y1 - 4.0,
            MAX_BLANK_GAP_PT
        )
        if trimmed_bot + 4.0 < cur_rect.y1 and (trimmed_bot - cur_rect.y0) >= 18.0:
            cur_rect = fitz.Rect(cur_rect.x0, cur_rect.y0, cur_rect.x1, trimmed_bot + 2.0)
    except Exception:
        pass
    return cur_rect

def detect_boxed_questions(page, gray, ink, W, H):
    k = int(110 * SC)
    m = ink.astype(np.uint8)
    h_lines = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(h_lines, connectivity=8)
    segs = []
    for i in range(1, n):
        x, y, w, h, _a = stats[i]
        if w >= k and h <= int(9 * SC):
            segs.append((x / SC, (y + h / 2.0) / SC, (x + w) / SC))
    segs.sort(key=lambda t: t[1])

    def side_cover(xc_lo, xc_hi, yt, yb):
        a, b = max(0, int(xc_lo * SC)), min(ink.shape[1], int(xc_hi * SC))
        r0, r1 = int((yt + 8) * SC), int((yb - 8) * SC)
        if b <= a or r1 <= r0:
            return 0.0, None
        win = ink[r0:r1, a:b]
        rows_hit = win.any(axis=1).mean()
        col = win.sum(axis=0)
        return float(rows_hit), (a + int(np.argmax(col))) / SC

    boxes = []
    used_bottom = set()
    for i, (ax0, ay, ax1) in enumerate(segs):
        for j in range(i + 1, len(segs)):
            bx0, by, bx1 = segs[j]
            dy = by - ay
            if dy < 24:
                continue
            if dy > 0.7 * H:
                break
            if abs(ax0 - bx0) > 14 or abs(ax1 - bx1) > 14:
                continue
            lcov, lx = side_cover(ax0 - 22, ax0 + 3, ay, by)
            rcov, rx = side_cover(ax1 - 3, ax1 + 22, ay, by)
            if lcov < 0.8 or rcov < 0.8 or j in used_bottom:
                continue
            rect = fitz.Rect(max(2.0, lx - 4), max(2.0, ay - 4), min(W - 2.0, rx + 4), min(H - 2.0, by + 4))
            if any((rect & r).get_area() > 0.8 * min(rect.get_area(), r.get_area()) for r in [b[0] for b in boxes]):
                break
            if box_has_question_number(page, gray, rect):
                trimmed_rect = trim_solution_inside_slide_box(page, gray, ink, rect)
                boxes.append((trimmed_rect,))
                used_bottom.add(j)
            break
    return _order_rects_reading([b[0] for b in boxes])

# ==============================================================================
# ৯. ক্যাশ-মুক্ত YOLO লোডার ও হাইব্রিড এক্সট্রাকশন পাইপলাইন
# ==============================================================================
def render_progress_card(placeholder, current_idx, total_pages, page_no, found_count, stage_label):
    pct = int(round((current_idx / max(1, total_pages)) * 100))
    placeholder.markdown(f"""
    <div class="progress-studio-card">
        <div class="ps-top-row">
            <div class="ps-status-text">
                <span class="ps-dot"></span>
                <span>{stage_label}</span>
            </div>
            <span class="ps-counter">{current_idx:02d}/{total_pages:02d} • {pct}%</span>
        </div>
        <div class="ps-track">
            <div class="ps-fill" style="width: {pct}%;"></div>
        </div>
        <div class="ps-sub-row">
            <span>PAGE #{page_no}</span>
            <span>FOUND: {found_count}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def _load_yolo_cached(abs_path, file_mtime, file_size):
    """ফাইলের মডিফিকেশন টাইম ও সাইজসহ লোড করে যেন নতুন ফাইল দিলে কখনো পুরোনো ক্যাশ আটকে না থাকে।"""
    from ultralytics import YOLO
    return YOLO(abs_path)

def get_active_yolo_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_model = os.path.join(base_dir, "question_detector.pt")

    # ফোল্ডারে মডেল না থাকলে Hugging Face থেকে সরাসরি ডাউনলোড করে নেবে
    if not os.path.exists(target_model):
        import requests
        url = "https://huggingface.co/bengaliAI/badlad-yolov8m-seg/resolve/main/best.pt"
        try:
            with st.spinner("✨ প্রথমবারের মতো 52MB এর শক্তিশালী বাংলা এআই (BaDLAD) মডেল ডাউনলোড হচ্ছে... দয়া করে একটু অপেক্ষা করুন ⏳"):
                resp = requests.get(url, stream=True)
                resp.raise_for_status()
                with open(target_model, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.toast("✅ মডেল ডাউনলোড সফল হয়েছে!")
        except Exception as e:
            return None, f"মডেল ডাউনলোডে ত্রুটি: {e}"

    try:
        mtime = os.path.getmtime(target_model)
        fsize = os.path.getsize(target_model)
        model = _load_yolo_cached(target_model, mtime, fsize)
        size_mb = fsize / (1024 * 1024)
        return model, f"সক্রিয় (BaDLAD Bengali AI • {size_mb:.1f} MB)"
    except Exception as e:
        return None, f"লোড ত্রুটি: {e}"

def extract_page_questions_with_yolo(yolo_model, page, ink_mask, W, H, pno, xray_logs=None):
    """ট্রেইন করা YOLOv8 ডিটেকশন এবং সেগমেন্টেশন (BaDLAD) উভয় মডেল সাপোর্ট করে।"""
    yolo_sc = 2.0
    pix = page.get_pixmap(matrix=fitz.Matrix(yolo_sc, yolo_sc), alpha=False)
    img_rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # BaDLAD মডেল অনেক বেশি বক্স দিতে পারে, তাই conf=0.25 রাখা হলো
    preds = yolo_model.predict(source=img_bgr, conf=0.25, iou=0.45, imgsz=1024, verbose=False)

    h_px, w_px = ink_mask.shape[:2]
    raw_boxes = []
    for r in preds:
        if r.boxes is None:
            continue
        for box in r.boxes:
            # সেগমেন্টেশন মাস্কের চারপাশের চতুর্ভুজ (Bounding box) বের করে আনা
            x0, y0, x1, y1 = [float(v) / yolo_sc for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # BaDLAD মডেলে 0=Paragraph, 1=Text Box, 2=Image, 3=Table
            # আমরা ছবি বা টেবিল (2, 3) এড়িয়ে যাব, আর আপনার পুরোনো মডেলের 0=question তো কাজ করবেই।
            if cls_id > 1:
                continue

            bw, bh = x1 - x0, y1 - y0

            if bw < 50.0 or bh < float(MIN_Q_HEIGHT_PT):
                continue
            if bw > W * 0.72 or bh > H * 0.55:
                continue
            if y0 < 30.0 or y1 > (H - 25.0):
                continue

            if x0 < (W * 0.42) and x1 > (W * 0.58):
                continue

            x0 = max(4.0, x0)
            y0 = max(4.0, y0)
            x1 = min(W - 4.0, x1 - RIGHT_EDGE_TRIM_PT)
            y1 = min(H - 4.0, y1)

            sub_ink = ink_mask[max(0, int(y0 * SC)):min(h_px, int(y1 * SC)), max(0, int(x0 * SC)):min(w_px, int(x1 * SC))]
            if sub_ink.size == 0 or float(sub_ink.mean()) < 0.008:
                continue

            y0, y1 = trim_vertical_whitespace(ink_mask, (x0, x1), y0, y1, MAX_BLANK_GAP_PT)
            if (x1 - x0) >= 50.0 and (y1 - y0) >= float(MIN_Q_HEIGHT_PT):
                raw_boxes.append((x0, y0, x1, y1, conf))

    if not raw_boxes:
        return []

    raw_boxes.sort(key=lambda b: b[4], reverse=True)
    kept = []
    for b in raw_boxes:
        x0, y0, x1, y1, conf = b
        bh = y1 - y0
        bw = x1 - x0
        is_dup = False
        for kb in kept:
            kx0, ky0, kx1, ky1, _ = kb
            inter_x = max(0.0, min(x1, kx1) - max(x0, kx0))
            inter_y = max(0.0, min(y1, ky1) - max(y0, ky0))
            if inter_x / max(1.0, min(bw, kx1 - kx0)) > 0.40 and inter_y / max(1.0, min(bh, ky1 - ky0)) > 0.30:
                is_dup = True
                break
        if not is_dup:
            kept.append(b)

    kept.sort(key=lambda b: (0 if ((b[0] + b[2]) * 0.5 < W * 0.5) else 1, b[1]))

    page_rects = []
    for idx_b, (x0, y0, x1, y1, conf) in enumerate(kept, 1):
        rect = fitz.Rect(x0, y0, x1, y1)
        if not rect.is_empty and not rect.is_infinite:
            page_rects.append(rect)
            if xray_logs is not None:
                xray_logs.append({
                    "পৃষ্ঠা": pno + 1,
                    "কলাম": 1 if (x0 + x1) * 0.5 < W * 0.5 else 2,
                    "Y": round(y0, 1),
                    "সিদ্ধান্ত": f"YOLO AI (#{idx_b} • {conf:.2f})",
                    "লাইন": f"Box: ({round(x0,1)}, {round(y0,1)}) -> ({round(x1,1)}, {round(y1,1)})"
                })

    return page_rects

def _merge_vision_and_yolo_rects(vision_rects, yolo_rects, W):
    """
    ভিশন ইঞ্জিন (ACS/উদ্ভাস) এবং YOLO AI মডেলের বক্সগুলোর সেরা সমন্বয় করে:
    ১. ভিশন ইঞ্জিন যেখানে নিখুঁত প্রশ্ন পেয়েছে সেগুলো রাখে।
    ২. যে প্রশ্নগুলো ভিশন ইঞ্জিন মিস করেছে কিন্তু YOLO ধরেছে, সেগুলোও যোগ করে!
    """
    if not vision_rects:
        return yolo_rects, len(yolo_rects), 0
    if not yolo_rects:
        return vision_rects, 0, len(vision_rects)

    merged = list(vision_rects)
    added_from_yolo = 0
    for yr in yolo_rects:
        y_col = 0 if (yr.x0 + yr.x1) * 0.5 < W * 0.5 else 1
        overlaps = False
        for vr in vision_rects:
            v_col = 0 if (vr.x0 + vr.x1) * 0.5 < W * 0.5 else 1
            if y_col == v_col:
                inter_y = max(0.0, min(yr.y1, vr.y1) - max(yr.y0, vr.y0))
                if inter_y / max(1.0, min(yr.height, vr.height)) > 0.25 or abs(yr.y0 - vr.y0) < 16.0:
                    overlaps = True
                    break
        if not overlaps:
            merged.append(yr)
            added_from_yolo += 1

    merged.sort(key=lambda r: (0 if (r.x0 + r.x1) * 0.5 < W * 0.5 else 1, r.y0))
    return merged, added_from_yolo, len(vision_rects)

def extract_questions_from_pdf(src_doc, page_indices, progress_slot, collect_xray=False):
    questions = []
    xray_logs = []
    total_to_scan = len(page_indices)

    yolo_model, yolo_status = get_active_yolo_model()
    yolo_total_hits = 0
    vision_total_hits = 0

    for idx, pno in enumerate(page_indices):
        render_progress_card(
            progress_slot, idx, total_to_scan, pno + 1, len(questions),
            f"পৃষ্ঠা {pno + 1} স্ক্যান চলছে..."
        )

        page = src_doc[pno]
        W, H = page.rect.width, page.rect.height
        gray = page_to_gray(page, SC)
        ink = get_ink_mask(gray)

        if is_slide_page(W, H):
            boxed = detect_boxed_questions(page, gray, ink, W, H)
            if boxed:
                for r in boxed:
                    questions.append((pno, r))
                render_progress_card(
                    progress_slot, idx + 1, total_to_scan, pno + 1, len(questions),
                    f"পৃষ্ঠা {pno + 1} সম্পন্ন"
                )
                continue

        log_target = xray_logs if collect_xray else None

        # ১. ভিশন ইঞ্জিন (উদ্ভাস + ACS) চালানো
        vision_rects = extract_page_questions_vision_hybrid(
            page, gray, ink, W, H, pno, xray_logs=log_target
        )

        # ২. ট্রেইনড YOLO AI মডেল চালানো
        yolo_rects = []
        if yolo_model is not None:
            yolo_rects = extract_page_questions_with_yolo(
                yolo_model, page, ink, W, H, pno, xray_logs=log_target
            )

        # ৩. দুটির ফিউশন (যাতে কোনো প্রশ্নই বাদ না যায়)
        page_rects, y_cnt, v_cnt = _merge_vision_and_yolo_rects(vision_rects, yolo_rects, W)
        yolo_total_hits += len(yolo_rects)
        vision_total_hits += v_cnt

        for r in page_rects:
            questions.append((pno, r))

        render_progress_card(
            progress_slot, idx + 1, total_to_scan, pno + 1, len(questions),
            f"পৃষ্ঠা {pno + 1} সম্পন্ন"
        )

    st.session_state["rv_xray_logs"] = xray_logs
    st.info(f"🧠 **YOLO মডেল:** {yolo_status} | **YOLO কাঁচা বক্স:** {yolo_total_hits}টি | **ভিশন ইঞ্জিন বক্স:** {vision_total_hits}টি | **চূড়ান্ত প্রশ্ন:** {len(questions)}টি")
    return questions

def build_practice_pdf(src_doc, questions, custom_gaps=None):
    if custom_gaps is None:
        custom_gaps = st.session_state.get("rv_custom_gaps", {})

    out = fitz.open()
    PW, PH = fitz.paper_size("a4")
    MARGIN_X, MARGIN_TOP, MARGIN_BOT, COL_GAP = 24.0, 50.0, 26.0, 18.0
    num_cols = int(COLUMNS_PER_PAGE)
    col_width = (PW - 2 * MARGIN_X - (num_cols - 1) * COL_GAP) / num_cols

    widths = sorted(c.width for _, c in questions)
    wide_set = widths[len(widths) // 2] > 1.5 * col_width
    common_scale = None
    if wide_set:
        num_cols = 1
        col_width = PW - 2 * MARGIN_X
        common_scale = min(1.65, col_width / max(widths))

    def create_styled_page():
        p = out.new_page(width=PW, height=PH)
        if HEADER_TITLE.strip():
            p.insert_text(fitz.Point(MARGIN_X, 30), HEADER_TITLE.strip(), fontsize=11, fontname="helv", color=(0.1, 0.1, 0.1))
            p.insert_text(fitz.Point(PW - MARGIN_X - 75, 30), f"Total Q: {len(questions)}", fontsize=10, fontname="helv", color=(0.2, 0.2, 0.2))
            p.draw_line(fitz.Point(MARGIN_X, 37), fitz.Point(PW - MARGIN_X, 37), color=(0.2, 0.2, 0.2), width=0.8)
        if num_cols == 2:
            mid_x = PW / 2.0
            p.draw_line(fitz.Point(mid_x, MARGIN_TOP - 6), fitz.Point(mid_x, PH - MARGIN_BOT), color=(0.3, 0.3, 0.3), width=0.6)
        return p

    page = create_styled_page()
    cur_col, y_cursor = 0, MARGIN_TOP
    avail_col_h = PH - MARGIN_TOP - MARGIN_BOT

    for q_idx, (pno, clip) in enumerate(questions):
        q_key = str(q_idx + 1)
        gap_pt = float(custom_gaps.get(q_key, custom_gaps.get(q_idx + 1, QUESTION_GAP_PT)))
        extra_space = max(0.0, gap_pt - float(QUESTION_GAP_PT))

        scale = common_scale if common_scale else min(col_width / clip.width, 1.35 if num_cols == 2 else 1.65)
        target_w, target_h = clip.width * scale, clip.height * scale
        needed_h = min(avail_col_h, target_h + extra_space)

        if y_cursor + needed_h > PH - MARGIN_BOT and y_cursor > MARGIN_TOP:
            cur_col += 1
            if cur_col >= num_cols:
                page = create_styled_page()
                cur_col = 0
            y_cursor = MARGIN_TOP

        x_start = MARGIN_X + cur_col * (col_width + COL_GAP)
        target_rect = fitz.Rect(x_start, y_cursor, x_start + target_w, y_cursor + target_h)
        if INVERT_DARK_OUT and clip_is_dark(src_doc[pno], clip):
            pix = src_doc[pno].get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip, alpha=False)
            pix.invert_irect(pix.irect)
            page.insert_image(target_rect, pixmap=pix)
        else:
            page.show_pdf_page(target_rect, src_doc, pno, clip=clip)
        y_cursor += target_h + gap_pt

    pdf_bytes = out.tobytes(garbage=4, deflate=True)
    out.close()
    return pdf_bytes

# ==============================================================================
# ৯.৯ বাহ্যিক canvas_editor.html লোডার ও বক্স এডিটর থেকে YOLO ডেটাসেট অটো-সেভার
# ==============================================================================
def save_corrected_boxes_to_yolo_dataset(raw_pdf, new_df):
    """বক্স এডিটরে সেভ চাপলে লোকাল ফোল্ডার এবং গিটহাবের yolo_dataset ফোল্ডারে সরাসরি সেভ করে।"""
    pushed_pages = 0
    push_errors = []
    try:
        img_dir = os.path.join("yolo_dataset", "images")
        lbl_dir = os.path.join("yolo_dataset", "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        doc = fitz.open(stream=raw_pdf, filetype="pdf")
        active_df = new_df[new_df["keep"] == True]
        for p_num, grp in active_df.groupby("page"):
            pno = int(p_num) - 1
            if not (0 <= pno < len(doc)) or len(grp) < 2:
                continue
            page = doc[pno]
            W, H = float(page.rect.width), float(page.rect.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            img_rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            yolo_lines = []
            for _, r in grp.iterrows():
                x0, y0, x1, y1 = float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])
                xc = max(0.0, min(1.0, ((x0 + x1) * 0.5) / W))
                yc = max(0.0, min(1.0, ((y0 + y1) * 0.5) / H))
                bw = max(0.01, min(1.0, abs(x1 - x0) / W))
                bh = max(0.01, min(1.0, abs(y1 - y0) / H))
                yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            stamp = f"edited_p{p_num}_{len(raw_pdf)}"
            lbl_text = "\n".join(yolo_lines) + "\n"

            # ১. লোকালে সেভ
            img_rel_path = f"yolo_dataset/images/{stamp}.jpg"
            lbl_rel_path = f"yolo_dataset/labels/{stamp}.txt"
            cv2.imwrite(img_rel_path, img_bgr)
            with open(lbl_rel_path, "w", encoding="utf-8") as f:
                f.write(lbl_text)

            # ২. সরাসরি গিটহাবে পুশ
            ok_jpg, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok_jpg:
                ok_img, msg_img = push_binary_file_to_github(img_rel_path, buf.tobytes(), f"Add training image {stamp}.jpg")
                if not ok_img:
                    push_errors.append(f"পৃষ্ঠা {p_num} ছবি: {msg_img}")
                    continue
                ok_lbl, msg_lbl = push_file_to_github(lbl_rel_path, lbl_text, f"Add training labels {stamp}.txt")
                if ok_lbl:
                    pushed_pages += 1
                else:
                    push_errors.append(f"পৃষ্ঠা {p_num} লেবেল: {msg_lbl}")
        doc.close()
        if pushed_pages > 0:
            st.toast(f"☁️ {pushed_pages}টি পৃষ্ঠার ডেটা সরাসরি গিটহাবে yolo_dataset-এ সেভ হয়েছে!")
        if push_errors:
            st.warning("⚠️ yolo_dataset গিটহাবে সেভ হয়নি: " + " | ".join(push_errors[:3]))
    except Exception as e:
        st.warning(f"⚠️ yolo_dataset সেভ ত্রুটি: {e}")

def _order_items_reading(items):
    items = sorted(items, key=lambda it: (it[0].y0, it[0].x0))
    rows, out = [], []
    for it in items:
        if rows and abs(it[0].y0 - rows[-1][0][0].y0) < 30:
            rows[-1].append(it)
        else:
            rows.append([it])
    for row in rows:
        out.extend(sorted(row, key=lambda it: it[0].x0))
    return out

def questions_to_rows(questions, w_flags=None):
    rows = []
    for idx, (pno, r) in enumerate(questions):
        is_w = bool(w_flags[idx]) if (w_flags and idx < len(w_flags)) else False
        rows.append({
            "keep": True,
            "page": pno + 1,
            "x0": round(float(r.x0), 1),
            "y0": round(float(r.y0), 1),
            "x1": round(float(r.x1), 1),
            "y1": round(float(r.y1), 1),
            "w_gap": is_w,
        })
    return pd.DataFrame(rows, columns=["keep", "page", "x0", "y0", "x1", "y1", "w_gap"])

def rows_to_questions(df, src_doc):
    per_page = {}
    for _, r in df.iterrows():
        try:
            if not bool(r.get("keep", True)):
                continue
            pno = int(r["page"]) - 1
            if not (0 <= pno < len(src_doc)):
                continue
            W, H = src_doc[pno].rect.width, src_doc[pno].rect.height
            y0, y1 = float(r["y0"]), float(r["y1"])
            if pd.isna(y0) or pd.isna(y1):
                continue
            x0 = 14.0 if pd.isna(r["x0"]) else float(r["x0"])
            x1 = W - 14.0 if pd.isna(r["x1"]) else float(r["x1"])
            rect = fitz.Rect(max(0.0, min(x0, x1)), max(0.0, min(y0, y1)), min(W, max(x0, x1)), min(H, max(y0, y1)))
            if rect.width < 18 or rect.height < 6:
                continue
            raw_w = r.get("w_gap", False)
            is_w = False if pd.isna(raw_w) else bool(raw_w)
            per_page.setdefault(pno, []).append((rect, is_w))
        except Exception:
            continue

    out = []
    custom_gaps = {}
    w_gap_pt = float(st.session_state.get("rv_w_gap_pt", 90.0))
    q_serial = 1

    for pno in sorted(per_page):
        W, H = src_doc[pno].rect.width, src_doc[pno].rect.height
        items = per_page[pno]
        if W > 1.2 * H:
            items = _order_items_reading(items)
        else:
            items = sorted(
                items,
                key=lambda it: (0 if (it[0].width > 0.6 * W or (it[0].x0 + it[0].x1) / 2 < W / 2) else 1, it[0].y0)
            )
        for rect, is_w in items:
            out.append((pno, rect))
            if is_w:
                custom_gaps[str(q_serial)] = w_gap_pt
            q_serial += 1

    st.session_state["rv_custom_gaps"] = custom_gaps
    return out

def apply_sidebar_batch_transform(df, raw_pdf, target_mode, box_nums_str, target_page, dx, dy, dw, dh):
    if df is None or df.empty:
        return df
    doc = fitz.open(stream=raw_pdf, filetype="pdf")
    ordered_qs = rows_to_questions(df, doc)
    custom_map = st.session_state.get("rv_custom_gaps", {})
    w_flags = [str(i + 1) in custom_map for i in range(len(ordered_qs))]
    ordered_df = questions_to_rows(ordered_qs, w_flags=w_flags)
    total_q = len(ordered_df)

    if target_mode == "নির্দিষ্ট বক্স":
        target_indices = parse_box_selection_string(box_nums_str, total_q)
    else:
        target_indices = set(range(total_q))

    updated_rows = []
    for idx, row in ordered_df.iterrows():
        pno = int(row["page"]) - 1
        W = doc[pno].rect.width if 0 <= pno < len(doc) else 595.0
        H = doc[pno].rect.height if 0 <= pno < len(doc) else 842.0
        x0, y0, x1, y1 = float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"])
        cx = (x0 + x1) * 0.5

        match = False
        if target_mode == "সব বক্স":
            match = True
        elif target_mode == "নির্দিষ্ট বক্স" and idx in target_indices:
            match = True
        elif target_mode == "বাম কলাম" and cx < (W * 0.5):
            match = True
        elif target_mode == "ডান কলাম" and cx >= (W * 0.5):
            match = True
        elif target_mode == "নির্দিষ্ট পেজ" and int(row["page"]) == int(target_page):
            match = True

        if match:
            x0 = max(2.0, min(W - 24.0, x0 + float(dx)))
            y0 = max(2.0, min(H - 18.0, y0 + float(dy)))
            x1 = max(x0 + 20.0, min(W - 2.0, x1 + float(dx) + float(dw)))
            y1 = max(y0 + 12.0, min(H - 2.0, y1 + float(dy) + float(dh)))

        updated_rows.append({
            "keep": True,
            "page": int(row["page"]),
            "x0": round(x0, 1),
            "y0": round(y0, 1),
            "x1": round(x1, 1),
            "y1": round(y1, 1),
            "w_gap": bool(row.get("w_gap", False)),
        })

    doc.close()
    return pd.DataFrame(updated_rows, columns=["keep", "page", "x0", "y0", "x1", "y1", "w_gap"])

def prepare_pages_payload_for_canvas(raw_pdf, page_indices):
    doc = fitz.open(stream=raw_pdf, filetype="pdf")
    pages_data = []
    for pno in page_indices:
        if not (0 <= pno < len(doc)):
            continue
        page = doc[pno]
        W, H = float(page.rect.width), float(page.rect.height)
        zoom = min(1.55, 920.0 / max(1.0, W))
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 1:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
        pages_data.append({
            "page": pno + 1,
            "w_pt": round(W, 2),
            "h_pt": round(H, 2),
            "img_b64": b64
        })
    doc.close()
    return pages_data

def build_interactive_canvas_html(pages_data, initial_boxes, file_key, template_file="canvas_editor.html"):
    pages_json = json.dumps(pages_data)
    boxes_json = json.dumps(initial_boxes)
    scan_ts = st.session_state.get("rv_scan_ts", "v5")
    safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", f"{file_key}_{scan_ts}")
    w_gap_pt = int(float(st.session_state.get("rv_w_gap_pt", 90)))

    if os.path.exists(template_file):
        with open(template_file, "r", encoding="utf-8") as f:
            tpl = f.read()
        return (
            tpl.replace("{{PAGES_JSON}}", pages_json)
               .replace("{{BOXES_JSON}}", boxes_json)
               .replace("{{SAFE_KEY}}", safe_key)
               .replace("{{W_GAP_PT}}", str(w_gap_pt))
        )
    return "<div style='color:#fff;padding:12px;'>canvas_editor.html ফাইলটি পাওয়া যায়নি।</div>"

def render_learning_suggestions():
    suggestions = st.session_state.get("rv_suggestions", [])
    if not suggestions:
        return

    total = len(suggestions)
    idx = st.session_state.get("rv_sug_idx", 0)
    if idx >= total:
        idx = max(0, total - 1)
    if idx < 0:
        idx = 0
    st.session_state["rv_sug_idx"] = idx

    sug = suggestions[idx]
    is_unlocked = st.session_state.get("is_admin_unlocked", False)

    type_map = {
        "question_tags": "প্রশ্ন ট্যাগ",
        "solution_tags": "সমাধান ট্যাগ",
        "header_noise_patterns": "জলছাপ"
    }
    rev_type_map = {v: k for k, v in type_map.items()}
    short_type = type_map.get(sug.get("type"), "প্রশ্ন ট্যাগ")
    disp_pat = sug.get("display_pattern", sug.get("pattern", ""))
    box_txt = f"বক্স #{sug['box_no']}" if sug.get("box_no") else f"পৃষ্ঠা {sug.get('page', 1)}"
    sample_txt = sug.get("sample", "")

    with st.container():
        st.markdown(f"""
        <div class="ps-sug-card">
            <div class="ps-sug-top">
                <div class="ps-sug-badges">
                    <span class="ps-sug-box">{box_txt}</span>
                    <span class="ps-sug-type">{short_type}</span>
                    <span class="ps-sug-pat">{disp_pat}</span>
                </div>
                <span class="ps-sug-count">{idx + 1} / {total}</span>
            </div>
            <div class="ps-sug-sample">{sample_txt}</div>
        </div>
        """, unsafe_allow_html=True)

        pin_val = ""
        if not is_unlocked:
            pin_val = st.text_input(
                "পিন",
                type="password",
                placeholder="🔒 পিন...",
                key=f"sug_pin_{idx}_{sug['pattern']}",
                label_visibility="collapsed"
            )

        sug_conflict = st.session_state.get("rv_sug_conflict")
        if sug_conflict and sug_conflict.get("pattern") == disp_pat:
            st.error(sug_conflict["err_text"])
            cat_opts = list(rev_type_map.keys())
            def_c_idx = cat_opts.index(short_type) if short_type in cat_opts else 0
            chosen_type_bn = st.selectbox(
                f"'{disp_pat}' কোন ক্যাটাগরিতে রাখতে চান?",
                cat_opts,
                index=def_c_idx,
                key=f"sug_conf_sel_{idx}_{disp_pat}"
            )
            mc1, mc2 = st.columns(2)
            with mc1:
                if st.button("✓ স্থানান্তর ও সেভ", key=f"sug_force_{idx}_{disp_pat}"):
                    target_cat = rev_type_map[chosen_type_bn]
                    ok_f, msg_f = commit_rule_to_kb(
                        target_cat,
                        disp_pat,
                        f"Moved and learned {target_cat} (Box #{sug.get('box_no', '?')})",
                        force_move=True
                    )
                    st.session_state.pop("rv_sug_conflict", None)
                    suggestions.pop(idx)
                    st.session_state["rv_suggestions"] = suggestions
                    st.session_state["rv_sug_idx"] = max(0, min(idx, len(suggestions) - 1))
                    if ok_f:
                        st.toast("স্থানান্তর ও সেভ হয়েছে!")
                    else:
                        st.warning(msg_f)
                    st.rerun()
            with mc2:
                if st.button("✕ বাতিল", key=f"sug_conf_cancel_{idx}_{disp_pat}"):
                    st.session_state.pop("rv_sug_conflict", None)
                    st.rerun()

        b1, b2, b3, b4, b5 = st.columns([1, 1, 1.6, 1.4, 1.4])
        with b1:
            if st.button("❮", key="sug_prev_btn", disabled=(idx <= 0), help="আগেরটি"):
                st.session_state.pop("rv_sug_conflict", None)
                st.session_state["rv_sug_idx"] = idx - 1
                st.rerun()
        with b2:
            if st.button("❯", key="sug_next_btn", disabled=(idx >= total - 1), help="পরেরটি"):
                st.session_state.pop("rv_sug_conflict", None)
                st.session_state["rv_sug_idx"] = idx + 1
                st.rerun()
        with b3:
            if st.button("✓ সেভ", key=f"sug_yes_{idx}_{sug['pattern']}", help="গিটহাবে সেভ"):
                if is_unlocked or verify_admin_pin(pin_val):
                    st.session_state["is_admin_unlocked"] = True
                    ok, msg = commit_rule_to_kb(
                        sug["type"],
                        disp_pat,
                        f"Learned {sug['type']} (Box #{sug.get('box_no', '?')})",
                        force_move=False
                    )
                    if ok:
                        st.session_state.pop("rv_sug_conflict", None)
                        suggestions.pop(idx)
                        st.session_state["rv_suggestions"] = suggestions
                        st.session_state["rv_sug_idx"] = max(0, min(idx, len(suggestions) - 1))
                        st.toast("সেভ হয়েছে!")
                        st.rerun()
                    elif msg.startswith("CONFLICT|"):
                        parts = msg.split("|", 2)
                        st.session_state["rv_sug_conflict"] = {
                            "pattern": disp_pat,
                            "err_text": parts[2]
                        }
                        st.rerun()
                    else:
                        st.warning(msg)
                else:
                    st.error("ভুল পিন!")
        with b4:
            if st.button("✕ বাদ", key=f"sug_no_{idx}_{sug['pattern']}", help="এটি বাদ দিন"):
                st.session_state.pop("rv_sug_conflict", None)
                suggestions.pop(idx)
                st.session_state["rv_suggestions"] = suggestions
                st.session_state["rv_sug_idx"] = max(0, min(idx, len(suggestions) - 1))
                st.rerun()
        with b5:
            if st.button("🗑 সব", key="sug_clear_all_btn", help="সব মুছুন"):
                st.session_state.pop("rv_sug_conflict", None)
                st.session_state["rv_suggestions"] = []
                st.session_state["rv_sug_idx"] = 0
                st.rerun()

def render_results_and_review(raw_pdf, file_key):
    ver = st.session_state.get("rv_ver", 0)

    bridge_val = st.text_input(
        "BOX_JSON_BRIDGE",
        value="",
        placeholder="BOX_JSON_BRIDGE_MARKER",
        key=f"bridge_in_{ver}",
        label_visibility="collapsed"
    )
    if bridge_val and bridge_val != st.session_state.get("last_bridge_val"):
        st.session_state["last_bridge_val"] = bridge_val
        try:
            old_df = st.session_state.get("rv_rows")
            payload = json.loads(bridge_val)
            if "w_gap_pt" in payload:
                st.session_state["rv_w_gap_pt"] = float(payload["w_gap_pt"])

            new_boxes = payload.get("boxes", [])
            for b in new_boxes:
                b["w_gap"] = bool(b.get("w_gap", False))
            new_df = pd.DataFrame(new_boxes, columns=["keep", "page", "x0", "y0", "x1", "y1", "w_gap"])

            save_corrected_boxes_to_yolo_dataset(raw_pdf, new_df)
            learned = analyze_user_corrections(raw_pdf, old_df, new_df)
            if learned:
                cur_sugs = st.session_state.get("rv_suggestions", [])
                existing_pats = {(s["type"], s["pattern"]) for s in cur_sugs}
                for item in learned:
                    if (item["type"], item["pattern"]) not in existing_pats:
                        cur_sugs.append(item)
                st.session_state["rv_suggestions"] = cur_sugs

            st.session_state["rv_rows"] = new_df
            src = fitz.open(stream=raw_pdf, filetype="pdf")
            qs = rows_to_questions(new_df, src)
            if qs:
                built_pdf = build_practice_pdf(src, qs)
                st.session_state["rv_auto_pdf"] = built_pdf
            else:
                st.session_state["rv_auto_pdf"] = None
            src.close()
        except Exception as e:
            st.error(f"ত্রুটি: {e}")

    render_learning_suggestions()

    rows_df = st.session_state["rv_rows"]
    active_count = len(rows_df[rows_df["keep"] == True]) if not rows_df.empty else 0

    if st.session_state.get("rv_auto_pdf") and active_count > 0:
        st.markdown(f"""
        <div class="result-banner">
            <div class="rb-left">
                <div class="rb-check">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                </div>
                <div>
                    <div class="rb-title">প্র্যাকটিস শিট প্রস্তুত</div>
                    <div class="rb-sub">{st.session_state.get("rv_pages", 0)} পৃষ্ঠা স্ক্যান সম্পন্ন</div>
                </div>
            </div>
            <div class="rb-stat-pill">{active_count}টি প্রশ্ন</div>
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            label="ডাউনলোড (PDF)",
            data=st.session_state["rv_auto_pdf"],
            file_name="Practice_Sheet.pdf",
            mime="application/pdf",
            key=f"dl_top_{ver}_{active_count}"
        )

    if XRAY_MODE and st.session_state.get("rv_xray_logs"):
        with st.popover("🔬 এক্স-রে ডায়াগনস্টিক রিপোর্ট দেখুন", use_container_width=True):
            st.dataframe(pd.DataFrame(st.session_state["rv_xray_logs"]), use_container_width=True)

    with st.expander("বক্স এডিটর", expanded=False):
        selected_indices = st.session_state.get("rv_page_indices", [0])
        pages_payload = prepare_pages_payload_for_canvas(raw_pdf, selected_indices)
        initial_boxes = rows_df.to_dict(orient="records") if not rows_df.empty else []

        if components is not None:
            canvas_html = build_interactive_canvas_html(pages_payload, initial_boxes, file_key)
            components.html(canvas_html, height=820, scrolling=False)

# ১০. মেইন ইন্টারফেস
st.markdown("""
<div class="studio-header">
    <div>
        <h1 class="studio-title">SMART SCANNER</h1>
        <p class="studio-sub">Cut Out Only Questions From PDF</p>
    </div>
    <span class="studio-badge">VISION v5.0</span>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("পিডিএফ ফাইল", type=["pdf"], label_visibility="collapsed")

if uploaded_file is not None:
    raw_pdf = uploaded_file.getvalue()
    size_mb = len(raw_pdf) / (1024 * 1024)
    try:
        tmp_doc = fitz.open(stream=raw_pdf, filetype="pdf")
        total_pdf_pages = len(tmp_doc)
        tmp_doc.close()
    except Exception:
        total_pdf_pages = 0

    st.markdown(f"""
    <div class="uploaded-file-card">
        <div class="uf-left">
            <div class="uf-icon-box">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#5E9E8B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
            </div>
            <div>
                <div class="uf-name">{uploaded_file.name}</div>
                <div class="uf-details">{size_mb:.2f} MB • {total_pdf_pages} PAGES</div>
            </div>
        </div>
        <div class="uf-badge">READY</div>
    </div>
    """, unsafe_allow_html=True)

    file_key = f"{uploaded_file.name}|{len(raw_pdf)}"
    if st.session_state.get("rv_file_key") != file_key:
        for _k in ("rv_rows", "rv_auto_pdf", "rv_pages", "rv_page_indices", "last_bridge_val", "rv_suggestions", "rv_xray_logs"):
            st.session_state.pop(_k, None)

    if st.button("স্ক্যান করুন"):
        progress_slot = st.empty()
        try:
            src_doc = fitz.open(stream=raw_pdf, filetype="pdf")
            selected_pages = parse_page_range(PAGE_RANGE_TEXT, len(src_doc))
            questions = extract_questions_from_pdf(src_doc, selected_pages, progress_slot, collect_xray=XRAY_MODE)
            progress_slot.empty()

            st.session_state["rv_rows"] = questions_to_rows(questions)
            st.session_state["rv_auto_pdf"] = build_practice_pdf(src_doc, questions) if questions else None
            st.session_state["rv_pages"] = len(selected_pages)
            st.session_state["rv_page_indices"] = selected_pages
            st.session_state["rv_file_key"] = file_key
            st.session_state["rv_scan_ts"] = str(int(time.time() * 1000))
            st.session_state["rv_suggestions"] = []
            st.session_state["rv_custom_gaps"] = {}
            st.session_state["rv_ver"] = st.session_state.get("rv_ver", 0) + 1
            src_doc.close()

            if not questions:
                st.warning("কোনো প্রশ্ন পাওয়া যায়নি। বক্স এডিটর থেকে বক্স যোগ করুন।")
        except Exception as e:
            progress_slot.empty()
            st.error(f"ত্রুটি: {e}")

    if APPLY_BATCH_BTN and st.session_state.get("rv_rows") is not None:
        try:
            updated_df = apply_sidebar_batch_transform(
                st.session_state["rv_rows"],
                raw_pdf,
                BATCH_TARGET,
                BATCH_BOX_NUMS,
                BATCH_PAGE_NO,
                BATCH_DX,
                BATCH_DY,
                BATCH_DW,
                BATCH_DH,
            )
            st.session_state["rv_rows"] = updated_df
            src_doc = fitz.open(stream=raw_pdf, filetype="pdf")
            qs = rows_to_questions(updated_df, src_doc)
            st.session_state["rv_auto_pdf"] = build_practice_pdf(src_doc, qs) if qs else None
            src_doc.close()
            st.session_state["rv_scan_ts"] = str(int(time.time() * 1000))
            st.session_state["rv_ver"] = st.session_state.get("rv_ver", 0) + 1
        except Exception as e:
            st.error(f"ত্রুটি: {e}")

    if st.session_state.get("rv_rows") is not None and st.session_state.get("rv_file_key") == file_key:
        render_results_and_review(raw_pdf, file_key)

# ১১. বাহ্যিক canvas_editor.html ফাইল থেকে গ্লোবাল ইউআই ও ডিভাইস মোড লোড করা
if components is not None:
    try:
        global_ui_html = build_interactive_canvas_html([], [], "GLOBAL_UI_SHELL")
        components.html(global_ui_html, height=0, scrolling=False)
    except Exception:
        pass

# ১২. নরম ঝাপসা ব্যাকগ্রাউন্ড ও সিন্থেসাইজ সাউন্ড এফেক্ট
def load_ambient_and_sfx(css_file="ambient.css", js_file="sfx.js"):
    try:
        load_external_css(css_file)
    except Exception:
        pass
    if components is None or not os.path.exists(js_file):
        return
    try:
        with open(js_file, "r", encoding="utf-8") as f:
            js_src = f.read()
        js_literal = json.dumps(js_src).replace("</", "<\\/")
        components.html(
            "<script>(function(){try{"
            "var P=window.parent,D=P.document;"
            "if(D.getElementById('ps-sfx-engine'))return;"
            "var s=D.createElement('script');s.id='ps-sfx-engine';"
            "s.textContent=" + js_literal + ";"
            "D.head.appendChild(s);"
            "}catch(e){}})();</script>",
            height=0,
            scrolling=False,
        )
    except Exception:
        pass

load_ambient_and_sfx()
