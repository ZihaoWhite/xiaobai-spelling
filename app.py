"""小白拼写：用完整拼写训练建立单词输入的肌肉记忆。"""

from __future__ import annotations

import difflib
import hashlib
import html
import os
import random
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote

import pandas as pd
import streamlit as st

from cloud_storage import (
    CloudStorageError,
    SupabaseConfig,
    SupabaseStorage,
    cloud_id_from_ref,
    cloud_ref,
    is_cloud_ref,
)
from learning_assistant import (
    LearningAssistantError,
    generate_learning_bundle,
    load_cached_bundle,
)


APP_NAME = "小白拼写"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "vocabulary_data"
UPLOAD_DIR = BASE_DIR / "uploaded_csv"
BACKUP_DIR = BASE_DIR / ".backups"
LEARNING_CACHE_DIR = BASE_DIR / "learning_cache"

REQUIRED_COLUMNS = (
    "单词",
    "中文释义",
    "类型",
    "当前状态",
    "当天答题次数",
    "当天正确",
    "当天错误",
)
TEXT_COLUMNS = ("单词", "中文释义", "类型")
NUMERIC_COLUMNS = ("当前状态", "当天答题次数", "当天正确", "当天错误")
PRACTICE_MODES = (
    "模式 A · 看英看中跟打",
    "模式 B · 看中文与首尾提示",
    "模式 C · 看中文盲拼",
)
PART_OF_SPEECH_PATTERN = re.compile(
    r"^\s*((?:prep|pron|conj|adj|adv|art|num|vt|vi|n|v)\.)\s*",
    re.IGNORECASE,
)


class CsvValidationError(ValueError):
    """A CSV exists but cannot be used as a vocabulary source."""

    def __init__(
        self,
        message: str,
        *,
        missing_columns: Sequence[str] | None = None,
        detected_columns: Sequence[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.missing_columns = list(missing_columns or [])
        self.detected_columns = list(detected_columns or [])


def choose_initial_sidebar_state(base_dir: Path = BASE_DIR) -> str:
    """Keep upload controls visible when a project has no CSV yet."""
    return "collapsed" if discover_csv_files(base_dir) else "expanded"


def configure_page() -> None:
    st.set_page_config(
        page_title=f"{APP_NAME} · 把单词练进手指里",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state=choose_initial_sidebar_state(BASE_DIR),
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --xb-ink: #0f172a;
            --xb-muted: #64748b;
            --xb-faint: #94a3b8;
            --xb-line: #e2e8f0;
            --xb-brand: #4f46e5;
            --xb-brand-dark: #4338ca;
            --xb-green: #16a34a;
            --xb-red: #dc2626;
            --xb-canvas: #f8fafc;
        }
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                "SF Pro Text", "Segoe UI", "PingFang SC", "Hiragino Sans GB",
                "Microsoft YaHei", sans-serif;
            color: var(--xb-ink);
        }
        .stApp {
            background:
                radial-gradient(circle at 72% 8%, rgba(99, 102, 241, .055), transparent 27rem),
                var(--xb-canvas);
        }
        [data-testid="stHeader"] { background: rgba(248, 250, 252, .86); }
        [data-testid="stToolbar"] { right: 1rem; }
        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 900px;
            padding-top: 1.65rem;
            padding-bottom: 4.5rem;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--xb-line);
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: .72rem;
        }
        .xb-brand { margin-bottom: 1rem; }
        .xb-brand-row {
            display: flex;
            align-items: center;
            gap: .75rem;
        }
        .xb-logo {
            display: grid;
            place-items: center;
            width: 46px;
            height: 46px;
            border-radius: 15px;
            background: linear-gradient(145deg, #4f46e5, #6366f1);
            border: 1px solid rgba(255, 255, 255, .52);
            color: var(--xb-brand);
            box-shadow: 0 8px 20px rgba(79, 70, 229, .2);
            color: #ffffff;
            font: 750 18px ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .xb-brand h1 {
            font-size: 1.62rem;
            letter-spacing: -.035em;
            line-height: 1.15;
            margin: 0;
        }
        .xb-brand p {
            color: var(--xb-muted);
            font-size: .92rem;
            margin: .25rem 0 0;
        }
        .xb-brand-row {
            justify-content: space-between;
        }
        .xb-brand-identity {
            display: flex;
            align-items: center;
            gap: .78rem;
        }
        .xb-keyboard-ready {
            display: inline-flex;
            align-items: center;
            gap: .48rem;
            color: #475569;
            background: rgba(255, 255, 255, .82);
            border: 1px solid var(--xb-line);
            border-radius: 999px;
            padding: .48rem .75rem;
            font-size: .78rem;
            box-shadow: 0 5px 18px rgba(15, 23, 42, .035);
        }
        .xb-keyboard-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #22c55e;
            box-shadow: 0 0 0 4px rgba(34, 197, 94, .12);
        }
        .xb-stat-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .72rem;
            margin-bottom: .7rem;
        }
        .xb-stat {
            padding: .85rem 1rem;
            background: rgba(255, 255, 255, .84);
            border: 1px solid var(--xb-line);
            border-radius: 15px;
        }
        .xb-stat-label {
            color: var(--xb-muted);
            font-size: .76rem;
            margin-bottom: .25rem;
        }
        .xb-stat-value {
            color: var(--xb-ink);
            font-size: 1.08rem;
            font-weight: 700;
            letter-spacing: -.02em;
        }
        .xb-progress {
            height: 7px;
            background: #e9edf5;
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 1.2rem;
        }
        .xb-progress > span {
            display: block;
            height: 100%;
            border-radius: inherit;
            background: var(--xb-brand);
            transition: width .2s ease;
        }
        .xb-session-rail {
            display: grid;
            grid-template-columns: auto minmax(120px, 1fr) auto;
            align-items: center;
            gap: 1.15rem;
            margin-bottom: 1rem;
            padding: .78rem 1rem;
            background: rgba(255, 255, 255, .78);
            border: 1px solid rgba(226, 232, 240, .95);
            border-radius: 16px;
            box-shadow: 0 7px 24px rgba(15, 23, 42, .035);
        }
        .xb-session-count {
            display: flex;
            align-items: baseline;
            gap: .22rem;
            min-width: 4.4rem;
        }
        .xb-session-count strong {
            font-size: 1.25rem;
            letter-spacing: -.04em;
        }
        .xb-session-count span {
            color: var(--xb-muted);
            font-size: .82rem;
        }
        .xb-session-track {
            height: 7px;
            overflow: hidden;
            background: #e9edf5;
            border-radius: 999px;
        }
        .xb-session-track span {
            display: block;
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #4f46e5, #818cf8);
            transition: width .2s ease;
        }
        .xb-session-metrics {
            display: flex;
            align-items: center;
            gap: .9rem;
            color: var(--xb-muted);
            font-size: .78rem;
            white-space: nowrap;
        }
        .xb-session-metrics strong {
            color: var(--xb-ink);
            font-size: .88rem;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border-color: var(--xb-line);
            border-radius: 26px;
            box-shadow: 0 18px 55px rgba(15, 23, 42, .07);
        }
        .xb-card-head {
            display: flex;
            align-items: center;
            gap: .48rem;
            justify-content: space-between;
            margin-bottom: 1.05rem;
        }
        .xb-card-labels {
            display: flex;
            align-items: center;
            gap: .48rem;
        }
        .xb-card-number {
            color: var(--xb-faint);
            font: 650 .72rem ui-monospace, SFMono-Regular, Menlo, monospace;
            letter-spacing: .08em;
        }
        .xb-pill {
            display: inline-flex;
            align-items: center;
            min-height: 26px;
            padding: 0 .68rem;
            border-radius: 999px;
            background: #eef2ff;
            color: #4338ca;
            font-size: .74rem;
            font-weight: 650;
            letter-spacing: .01em;
        }
        .xb-pill-neutral {
            background: #f1f5f9;
            color: #475569;
        }
        .xb-word {
            color: var(--xb-ink);
            font: 700 clamp(3.15rem, 7vw, 4rem)/1.08 ui-monospace,
                SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            letter-spacing: -.055em;
            overflow-wrap: anywhere;
            text-align: center;
            margin: .55rem 0 1rem;
        }
        .xb-hint {
            color: #334155;
            font: 650 clamp(2rem, 5vw, 2.85rem)/1.3 ui-monospace,
                SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            letter-spacing: .045em;
            overflow-wrap: anywhere;
            text-align: center;
            margin: .8rem 0 1.15rem;
        }
        .xb-hidden-word {
            color: var(--xb-faint);
            font-size: .9rem;
            letter-spacing: .08em;
            text-transform: uppercase;
            text-align: center;
            margin: 1.5rem 0 1.1rem;
        }
        .xb-meaning {
            color: #334155;
            font-size: clamp(1.12rem, 2.4vw, 1.34rem);
            line-height: 1.7;
            text-align: center;
            max-width: 710px;
            margin: 0 auto .6rem;
        }
        .xb-pos {
            color: var(--xb-brand);
            font: 650 .82rem ui-monospace, SFMono-Regular, Menlo, monospace;
            text-align: center;
            margin-bottom: 1.15rem;
        }
        .xb-audio-label {
            color: var(--xb-muted);
            font-size: .78rem;
            margin: .85rem 0 .2rem;
            text-align: center;
        }
        [data-testid="stAudio"] {
            max-width: 420px;
            margin: 0 auto .25rem;
        }
        .xb-input-guide {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: .46rem;
            margin: .55rem 0 .42rem;
            color: var(--xb-muted);
            font-size: .78rem;
        }
        .xb-input-guide strong {
            color: #334155;
            font-weight: 650;
        }
        .xb-input-guide kbd {
            display: inline-grid;
            place-items: center;
            min-width: 3.3rem;
            height: 1.72rem;
            padding: 0 .46rem;
            border: 1px solid #cbd5e1;
            border-bottom-width: 2px;
            border-radius: 7px;
            color: #475569;
            background: #ffffff;
            font: 650 .7rem ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        [data-testid="stTextInput"] input {
            min-height: 64px;
            border-color: #cbd5e1;
            border-radius: 13px;
            font: 650 1.45rem ui-monospace, SFMono-Regular, Menlo, Monaco,
                Consolas, monospace;
            letter-spacing: .02em;
            padding-inline: 1rem;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: var(--xb-brand);
            box-shadow: 0 0 0 4px rgba(79, 70, 229, .12);
        }
        [data-testid="stFormSubmitButton"] button,
        div[data-testid="stButton"] button[kind="primary"] {
            min-height: 46px;
            border-radius: 12px;
            font-weight: 680;
            background: var(--xb-brand);
            border-color: var(--xb-brand);
        }
        [data-testid="stFormSubmitButton"] button:hover,
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background: var(--xb-brand-dark);
            border-color: var(--xb-brand-dark);
        }
        div[data-testid="stButton"] button {
            border-radius: 10px;
        }
        .xb-feedback {
            border-radius: 14px;
            padding: .92rem 1rem;
            margin-top: .65rem;
            font-weight: 650;
        }
        .xb-feedback-ok {
            color: #166534;
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
        }
        .xb-feedback-bad {
            color: #991b1b;
            background: #fff7f7;
            border: 1px solid #fecaca;
        }
        .xb-recent {
            margin-bottom: .85rem;
            padding: .72rem .9rem;
            border-radius: 14px;
            border: 1px solid;
            background: rgba(255, 255, 255, .82);
            box-shadow: 0 6px 20px rgba(15, 23, 42, .03);
        }
        .xb-recent-main {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: .75rem;
        }
        .xb-recent-title {
            display: flex;
            align-items: center;
            gap: .55rem;
            font-size: .86rem;
            font-weight: 680;
        }
        .xb-recent-answer {
            color: #475569;
            font: 650 .84rem ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .xb-recent-ok {
            border-color: #bbf7d0;
            background: rgba(240, 253, 244, .88);
            color: #166534;
        }
        .xb-recent-bad {
            border-color: #fecaca;
            background: rgba(255, 247, 247, .9);
            color: #991b1b;
        }
        .xb-recent .xb-diff {
            padding-top: .65rem;
        }
        .xb-recent-warning {
            margin-top: .55rem;
            padding-top: .5rem;
            border-top: 1px solid rgba(180, 83, 9, .16);
            color: #92400e;
            font-size: .76rem;
            line-height: 1.55;
        }
        .xb-answer-pair {
            color: var(--xb-muted);
            font-size: .88rem;
            line-height: 1.7;
            margin: .85rem 0 .7rem;
        }
        .xb-answer-pair code {
            color: var(--xb-ink);
            background: #f1f5f9;
            border-radius: 6px;
            padding: .12rem .38rem;
        }
        .xb-diff {
            overflow-x: auto;
            padding: .85rem 0 .15rem;
        }
        .xb-diff-row {
            display: flex;
            align-items: center;
            width: max-content;
            min-width: 100%;
            margin-bottom: .38rem;
        }
        .xb-diff-label {
            flex: 0 0 5rem;
            color: var(--xb-muted);
            font-size: .76rem;
        }
        .xb-char {
            display: inline-grid;
            place-items: center;
            width: 1.72rem;
            height: 2rem;
            margin-right: 3px;
            border-radius: 6px;
            font: 650 1rem ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .xb-char-ok { color: #166534; background: #dcfce7; }
        .xb-char-bad { color: #b91c1c; background: #fee2e2; }
        .xb-char-extra {
            color: #b91c1c;
            background: #fee2e2;
            text-decoration: line-through;
        }
        .xb-char-missing {
            color: #94a3b8;
            background: #f1f5f9;
            text-decoration: underline;
            text-underline-offset: 3px;
        }
        .xb-char-blank { color: transparent; background: transparent; }
        .xb-complete {
            text-align: center;
            padding: 1.4rem .5rem .8rem;
        }
        .xb-complete-mark {
            display: grid;
            place-items: center;
            width: 58px;
            height: 58px;
            border-radius: 18px;
            background: #ecfdf5;
            color: var(--xb-green);
            font-size: 1.7rem;
            margin: 0 auto 1rem;
        }
        .xb-complete h2 {
            font-size: 1.85rem;
            letter-spacing: -.035em;
            margin: 0 0 .45rem;
        }
        .xb-complete p { color: var(--xb-muted); margin: 0; }
        .xb-empty {
            text-align: center;
            padding: 3rem 1rem;
        }
        .xb-empty h2 { font-size: 1.5rem; margin-bottom: .5rem; }
        .xb-empty p { color: var(--xb-muted); line-height: 1.75; }
        .xb-side-section {
            color: var(--xb-muted);
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin: .8rem 0 -.25rem;
        }
        .xb-side-brand {
            display: flex;
            align-items: center;
            gap: .62rem;
            margin: .2rem 0 .9rem;
            padding-bottom: .9rem;
            border-bottom: 1px solid #eef2f7;
        }
        .xb-side-mark {
            display: grid;
            place-items: center;
            width: 34px;
            height: 34px;
            border-radius: 11px;
            color: #ffffff;
            background: #4f46e5;
            font: 750 .76rem ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .xb-side-brand strong {
            display: block;
            color: var(--xb-ink);
            font-size: .95rem;
        }
        .xb-side-brand span {
            display: block;
            color: var(--xb-muted);
            font-size: .69rem;
            margin-top: .05rem;
        }
        .xb-file-meta {
            background: #f8fafc;
            border: 1px solid var(--xb-line);
            border-radius: 12px;
            padding: .75rem .8rem;
            color: #475569;
            font-size: .78rem;
            line-height: 1.65;
            overflow-wrap: anywhere;
        }
        .xb-save-ok { color: #15803d; }
        .xb-save-idle { color: #64748b; }
        .xb-save-bad { color: #b91c1c; }
        @media (max-width: 720px) {
            [data-testid="stAppViewContainer"] > .main .block-container {
                padding-top: 1.35rem;
                padding-inline: 1rem;
            }
            .xb-keyboard-ready { display: none; }
            .xb-session-rail {
                grid-template-columns: auto 1fr;
                gap: .7rem;
            }
            .xb-session-metrics {
                grid-column: 1 / -1;
                justify-content: space-between;
            }
            .xb-stat-grid { grid-template-columns: repeat(2, 1fr); }
            .xb-stat { padding: .72rem .82rem; }
            .xb-word { font-size: clamp(2.25rem, 12vw, 2.8rem); }
            .xb-hint { font-size: clamp(1.65rem, 8vw, 2.15rem); }
            .xb-char { width: 1.48rem; height: 1.8rem; }
        }

        /* 2026 refresh: a calm study-desk interface */
        :root {
            --xb-ink: #15342d;
            --xb-muted: #6c7d78;
            --xb-faint: #98aaa5;
            --xb-line: #dce8e3;
            --xb-brand: #109b78;
            --xb-brand-dark: #08765c;
            --xb-green: #109b78;
            --xb-red: #d85f4b;
            --xb-canvas: #f2f6f1;
            --xb-warm: #fffdf8;
            --xb-coral: #ff8063;
        }
        html, body, [class*="css"] {
            color: var(--xb-ink);
        }
        .stApp {
            background:
                radial-gradient(circle at 12% 4%, rgba(167, 229, 205, .36), transparent 27rem),
                radial-gradient(circle at 92% 12%, rgba(255, 211, 159, .26), transparent 25rem),
                linear-gradient(180deg, #f8faf6 0%, var(--xb-canvas) 64%, #eef4ef 100%);
        }
        [data-testid="stHeader"] {
            height: 60px !important;
            min-height: 60px !important;
            overflow: visible !important;
            pointer-events: auto;
            background: transparent !important;
        }
        [data-testid="stDecoration"] {
            display: none;
        }
        [data-testid="stToolbar"] {
            display: flex !important;
            pointer-events: auto;
            background: transparent !important;
        }
        [data-testid="stHeaderActionElements"] {
            display: flex !important;
            pointer-events: auto;
        }
        [data-testid="stMainMenu"],
        [data-testid="stAppDeployButton"],
        [data-testid="stStatusWidget"] {
            display: none !important;
        }
        [data-testid="stExpandSidebarButton"] {
            display: block !important;
            position: fixed !important;
            top: 1rem;
            left: 1rem;
            z-index: 20;
            pointer-events: auto;
            width: 42px !important;
            height: 42px !important;
            padding: 0 !important;
            border: 1px solid rgba(21, 52, 45, .08);
            border-radius: 14px;
            color: var(--xb-ink);
            background: rgba(255, 255, 255, .82);
            box-shadow: 0 8px 30px rgba(28, 65, 55, .08);
            backdrop-filter: blur(16px);
        }
        [data-testid="stExpandSidebarButton"]:hover {
            color: var(--xb-brand-dark);
            border-color: rgba(16, 155, 120, .24);
            background: #ffffff;
        }
        [data-testid="stSidebarCollapseButton"] {
            pointer-events: auto !important;
        }
        [data-testid="stMainBlockContainer"] {
            width: 100%;
            max-width: 980px;
            margin: 0 auto;
            padding: 1.2rem 1.5rem 4rem !important;
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(246, 251, 248, .98), rgba(237, 246, 241, .98));
            border-right: 1px solid rgba(22, 101, 80, .1);
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: .62rem;
        }
        .xb-brand {
            margin: 0 0 1rem;
            padding: 0 .15rem;
        }
        .xb-brand-row {
            justify-content: space-between;
        }
        .xb-brand-identity {
            gap: .85rem;
        }
        .xb-logo {
            position: relative;
            width: 50px;
            height: 50px;
            border-radius: 17px;
            background:
                radial-gradient(circle at 72% 24%, rgba(255, 255, 255, .58), transparent 22%),
                linear-gradient(145deg, #13aa83, #08765c);
            border: 1px solid rgba(255, 255, 255, .72);
            box-shadow:
                0 12px 26px rgba(16, 155, 120, .2),
                inset 0 1px 0 rgba(255, 255, 255, .34);
            color: #ffffff;
            font: 760 1.3rem/1 ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .xb-logo::after {
            content: "";
            position: absolute;
            right: 6px;
            bottom: 6px;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #ffd17d;
            box-shadow: 0 0 0 3px rgba(255, 255, 255, .25);
        }
        .xb-brand h1 {
            color: var(--xb-ink);
            font-size: 1.55rem;
            font-weight: 770;
            letter-spacing: -.045em;
        }
        .xb-brand p {
            color: var(--xb-muted);
            font-size: .82rem;
            margin-top: .18rem;
        }
        .xb-keyboard-ready {
            gap: .55rem;
            padding: .52rem .8rem;
            color: #45645b;
            background: rgba(255, 255, 255, .66);
            border: 1px solid rgba(74, 125, 109, .14);
            box-shadow: none;
            backdrop-filter: blur(14px);
        }
        .xb-keyboard-dot {
            background: #11a67e;
            box-shadow: 0 0 0 4px rgba(17, 166, 126, .12);
        }
        .xb-session-rail {
            grid-template-columns: auto minmax(170px, 1fr) auto;
            gap: 1.35rem;
            margin-bottom: 1rem;
            padding: .82rem 1.05rem;
            background: rgba(255, 255, 255, .62);
            border: 1px solid rgba(91, 135, 122, .13);
            border-radius: 20px;
            box-shadow: 0 10px 36px rgba(30, 75, 62, .06);
            backdrop-filter: blur(18px);
        }
        .xb-session-count {
            min-width: 5.5rem;
            gap: .25rem;
        }
        .xb-session-count strong {
            color: var(--xb-ink);
            font-size: 1.35rem;
            font-weight: 780;
        }
        .xb-session-count span {
            font-size: .8rem;
        }
        .xb-session-track {
            height: 8px;
            background: #deebe5;
        }
        .xb-session-track span {
            background: linear-gradient(90deg, #0b8e6d, #32bd91);
            box-shadow: 0 0 14px rgba(16, 155, 120, .18);
        }
        .xb-session-metrics {
            gap: .55rem;
        }
        .xb-metric {
            display: flex;
            align-items: baseline;
            gap: .3rem;
            padding: .38rem .58rem;
            border-radius: 10px;
            background: rgba(240, 247, 243, .82);
        }
        .xb-metric span {
            color: var(--xb-muted);
            font-size: .68rem;
        }
        .xb-metric strong {
            color: var(--xb-ink);
            font-size: .86rem;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            overflow: hidden;
            background:
                radial-gradient(circle at 82% 2%, rgba(255, 218, 173, .22), transparent 19rem),
                var(--xb-warm);
            border: 1px solid rgba(62, 112, 97, .14);
            border-radius: 30px;
            box-shadow:
                0 24px 70px rgba(31, 72, 60, .11),
                0 2px 8px rgba(31, 72, 60, .04);
        }
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 1.2rem 1.35rem 1.35rem;
        }
        .xb-card-head {
            margin-bottom: .72rem;
        }
        .xb-pill {
            min-height: 28px;
            padding: 0 .72rem;
            color: #08765c;
            background: #ddf4ea;
            border: 1px solid rgba(16, 155, 120, .08);
        }
        .xb-pill-neutral {
            color: #8d5b25;
            background: #fff0d7;
            border-color: rgba(190, 126, 46, .08);
        }
        .xb-card-number {
            color: #8aa098;
            letter-spacing: .12em;
        }
        .xb-prompt-zone {
            position: relative;
            overflow: hidden;
            padding: 1.45rem 1.25rem 1.25rem;
            margin-bottom: .85rem;
            border: 1px solid rgba(91, 135, 122, .11);
            border-radius: 22px;
            background:
                linear-gradient(145deg, rgba(234, 247, 240, .88), rgba(249, 250, 241, .84));
        }
        .xb-prompt-zone::before {
            content: "SPELL";
            position: absolute;
            right: 1.1rem;
            top: .8rem;
            color: rgba(16, 104, 81, .08);
            font: 800 2.8rem/1 ui-monospace, SFMono-Regular, Menlo, monospace;
            letter-spacing: -.08em;
        }
        .xb-word {
            position: relative;
            color: #153d34;
            font-size: clamp(3rem, 7vw, 4.55rem);
            font-weight: 760;
            letter-spacing: -.065em;
            margin: .25rem 0 .75rem;
            text-shadow: 0 2px 0 rgba(255, 255, 255, .7);
        }
        .xb-hint {
            position: relative;
            color: #28554a;
            font-size: clamp(2.1rem, 5vw, 3.2rem);
            margin: .55rem 0 .9rem;
        }
        .xb-hidden-word {
            position: relative;
            color: #6f8a82;
            font-size: .78rem;
            font-weight: 720;
            letter-spacing: .16em;
            margin: 1.15rem 0 .95rem;
        }
        .xb-meaning {
            position: relative;
            color: #3c5e55;
            font-size: clamp(1.08rem, 2.2vw, 1.28rem);
            line-height: 1.65;
            margin-bottom: .38rem;
        }
        .xb-pos {
            position: relative;
            display: table;
            margin: .4rem auto 0;
            padding: .2rem .5rem;
            color: #0b8467;
            background: rgba(255, 255, 255, .72);
            border-radius: 8px;
        }
        .xb-audio-label {
            color: #71877f;
            font-size: .72rem;
            letter-spacing: .03em;
            margin: .55rem 0 .2rem;
        }
        [data-testid="stAudio"] {
            max-width: 360px;
            margin-bottom: .15rem;
            filter: saturate(.72);
        }
        .xb-input-guide {
            margin: .45rem 0 .5rem;
            color: #71817d;
        }
        .xb-input-guide strong {
            color: #2c5147;
        }
        .xb-input-guide kbd {
            height: 1.65rem;
            color: #0b7d62;
            background: #eff8f4;
            border-color: #b9d8cc;
            box-shadow: 0 2px 0 rgba(22, 101, 80, .08);
        }
        [data-testid="stTextInput"] input {
            min-height: 66px;
            color: #153d34;
            caret-color: var(--xb-brand);
            background: #ffffff;
            border: 1.5px solid #cdded7;
            border-radius: 16px;
            font-size: 1.42rem;
            box-shadow: inset 0 1px 2px rgba(28, 65, 55, .025);
        }
        [data-testid="stTextInput"] input::placeholder {
            color: #a1b1ac;
            font-weight: 540;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: var(--xb-brand);
            box-shadow:
                0 0 0 4px rgba(16, 155, 120, .12),
                0 8px 22px rgba(16, 155, 120, .06);
        }
        [data-testid="stFormSubmitButton"] button,
        div[data-testid="stButton"] button[kind="primary"] {
            min-height: 50px;
            border: 0;
            border-radius: 14px;
            color: #ffffff;
            background: linear-gradient(135deg, #109b78, #0a8063);
            box-shadow: 0 10px 20px rgba(16, 155, 120, .16);
        }
        [data-testid="stFormSubmitButton"] button:hover,
        div[data-testid="stButton"] button[kind="primary"]:hover {
            border: 0;
            background: linear-gradient(135deg, #0c8b6b, #076e56);
            box-shadow: 0 12px 24px rgba(16, 155, 120, .22);
            transform: translateY(-1px);
        }
        div[data-testid="stButton"] button {
            border-color: #cdded7;
            border-radius: 12px;
        }
        .xb-recent {
            border-radius: 16px;
            box-shadow: 0 8px 22px rgba(31, 72, 60, .04);
        }
        .xb-recent-ok {
            color: #08765c;
            border-color: #bce4d4;
            background: rgba(234, 248, 241, .92);
        }
        .xb-recent-bad {
            color: #a54838;
            border-color: #f1c6bc;
            background: rgba(255, 243, 238, .94);
        }
        .xb-feedback-ok {
            color: #08765c;
            background: #eaf8f1;
            border-color: #bce4d4;
        }
        .xb-feedback-bad {
            color: #a54838;
            background: #fff3ee;
            border-color: #f1c6bc;
        }
        .xb-side-brand {
            margin-top: .45rem;
            border-bottom-color: rgba(22, 101, 80, .1);
        }
        .xb-side-mark {
            border-radius: 12px;
            background: linear-gradient(145deg, #13aa83, #08765c);
            box-shadow: 0 8px 18px rgba(16, 155, 120, .18);
        }
        .xb-file-meta {
            color: #45645b;
            background: rgba(255, 255, 255, .64);
            border-color: rgba(74, 125, 109, .14);
        }
        .xb-complete-mark {
            color: #08765c;
            background: #ddf4ea;
        }
        .xb-stat {
            background: rgba(239, 248, 244, .76);
            border-color: rgba(74, 125, 109, .12);
        }
        .xb-ai-shell {
            margin-top: 1rem;
            padding: 1rem 1.05rem;
            border: 1px solid rgba(74, 125, 109, .14);
            border-radius: 20px;
            background:
                radial-gradient(circle at 92% 0%, rgba(255, 210, 125, .17), transparent 14rem),
                rgba(255, 253, 248, .78);
            box-shadow: 0 10px 34px rgba(31, 72, 60, .055);
        }
        .xb-ai-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: .8rem;
        }
        .xb-ai-title {
            display: flex;
            align-items: center;
            gap: .55rem;
            color: var(--xb-ink);
            font-weight: 760;
        }
        .xb-ai-spark {
            display: inline-grid;
            place-items: center;
            width: 30px;
            height: 30px;
            border-radius: 10px;
            color: #08765c;
            background: #ddf4ea;
        }
        .xb-ai-subtitle {
            color: var(--xb-muted);
            font-size: .76rem;
            margin-top: .2rem;
        }
        .xb-ai-model {
            flex: 0 0 auto;
            max-width: 260px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            padding: .3rem .55rem;
            border-radius: 999px;
            color: #6d5a36;
            background: #fff1d9;
            font: 650 .66rem ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .xb-ai-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: .7rem;
        }
        .xb-ai-item {
            min-height: 90px;
            padding: .82rem .9rem;
            border: 1px solid rgba(74, 125, 109, .1);
            border-radius: 14px;
            background: rgba(255, 255, 255, .7);
        }
        .xb-ai-item-wide {
            grid-column: 1 / -1;
        }
        .xb-ai-label {
            color: #799089;
            font-size: .67rem;
            font-weight: 730;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: .32rem;
        }
        .xb-ai-value {
            color: #34584e;
            font-size: .86rem;
            line-height: 1.65;
        }
        .xb-ai-example {
            color: #153d34;
            font: 650 .98rem/1.6 ui-monospace, SFMono-Regular, Menlo, monospace;
            margin-bottom: .25rem;
        }
        .xb-ai-translation {
            color: #667d76;
            font-size: .82rem;
        }
        .xb-ai-origin {
            margin-top: .7rem;
            padding: .75rem .85rem;
            border-radius: 13px;
            color: #5d6f6a;
            background: rgba(238, 246, 242, .82);
            font-size: .76rem;
            line-height: 1.6;
        }
        .xb-ai-origin strong {
            color: #31584d;
        }
        .xb-family {
            display: flex;
            flex-wrap: wrap;
            gap: .35rem;
            margin-top: .7rem;
        }
        .xb-family span {
            padding: .25rem .48rem;
            border-radius: 8px;
            color: #0b7d62;
            background: #e8f6f0;
            font: 650 .72rem ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .xb-ai-idle {
            color: var(--xb-muted);
            font-size: .82rem;
            line-height: 1.7;
            margin-bottom: .7rem;
        }
        .xb-ai-status {
            display: flex;
            align-items: center;
            gap: .5rem;
            padding: .65rem .72rem;
            border: 1px solid rgba(74, 125, 109, .13);
            border-radius: 12px;
            color: #45645b;
            background: rgba(255, 255, 255, .58);
            font-size: .75rem;
        }
        .xb-ai-status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #10a37f;
            box-shadow: 0 0 0 4px rgba(16, 163, 127, .1);
        }
        .xb-ai-status-off .xb-ai-status-dot {
            background: #d97706;
            box-shadow: 0 0 0 4px rgba(217, 119, 6, .1);
        }
        @media (max-width: 720px) {
            [data-testid="stMainBlockContainer"] {
                padding-top: 1rem !important;
                padding-inline: .85rem;
            }
            .xb-brand {
                padding-left: 3.25rem;
            }
            .xb-logo {
                width: 44px;
                height: 44px;
                border-radius: 15px;
            }
            .xb-brand h1 { font-size: 1.35rem; }
            .xb-brand p { font-size: .74rem; }
            .xb-keyboard-ready { display: none; }
            .xb-session-rail {
                grid-template-columns: auto 1fr;
                gap: .68rem;
                border-radius: 17px;
            }
            .xb-session-metrics {
                grid-column: 1 / -1;
                justify-content: space-between;
            }
            .xb-metric {
                flex: 1;
                justify-content: center;
            }
            [data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: 22px;
            }
            [data-testid="stVerticalBlockBorderWrapper"] > div {
                padding: .9rem .9rem 1rem;
            }
            .xb-prompt-zone {
                padding: 1.2rem .8rem 1rem;
                border-radius: 18px;
            }
            .xb-prompt-zone::before {
                font-size: 2rem;
            }
            .xb-ai-grid {
                grid-template-columns: 1fr;
            }
            .xb-ai-item-wide {
                grid-column: auto;
            }
            .xb-ai-model {
                display: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def parse_date_from_filename(filename: str) -> str | None:
    """Return an ISO date from YYYY-MM-DD or YYYYMMDD in a filename."""
    match = re.search(
        r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)",
        Path(filename).stem,
    )
    if not match:
        return None
    year, month, day = map(int, match.groups())
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def make_file_record(path: Path, base_dir: Path = BASE_DIR) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    try:
        parent_label = resolved.parent.relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        parent_label = resolved.parent.name
    if parent_label == ".":
        parent_label = "项目根目录"
    return {
        "path": str(resolved),
        "name": resolved.name,
        "parent": parent_label,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "date": parse_date_from_filename(resolved.name),
        "missing": False,
    }


def sort_file_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put export files first, then sort by modification time descending."""
    return sorted(
        records,
        key=lambda record: (
            0 if "export" in str(record["name"]).casefold() else 1,
            -float(record["mtime"]),
            str(record["path"]).casefold(),
        ),
    )


def discover_csv_files(base_dir: Path = BASE_DIR) -> list[dict[str, Any]]:
    """Scan only the three documented locations and return file metadata."""
    directories = (base_dir, base_dir / "vocabulary_data", base_dir / "uploaded_csv")
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.glob("*.csv"):
            if path.name.startswith(".") or ".tmp" in path.name.casefold():
                continue
            try:
                record = make_file_record(path, base_dir)
            except (OSError, ValueError):
                continue
            if record["path"] in seen:
                continue
            seen.add(record["path"])
            records.append(record)
    return sort_file_records(records)


def build_file_label(
    record: dict[str, Any], records: Sequence[dict[str, Any]] | None = None
) -> str:
    duplicate_name = (
        sum(item["name"] == record["name"] for item in (records or [])) > 1
    )
    location = f"{record['parent']} / " if duplicate_name else ""
    date_label = record.get("date")
    if not date_label:
        date_label = datetime.fromtimestamp(float(record["mtime"])).strftime("%Y-%m-%d")
    missing_suffix = " · 文件已移除" if record.get("missing") else ""
    return f"{date_label} · {location}{record['name']}{missing_suffix}"


def file_signature(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def read_csv_safely(path: Path) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc.reason}")
        except pd.errors.EmptyDataError as exc:
            raise CsvValidationError("CSV 文件为空，未检测到表头。") from exc
        except pd.errors.ParserError as exc:
            raise CsvValidationError(f"CSV 格式无法解析：{exc}") from exc
        except OSError as exc:
            raise CsvValidationError(f"无法读取文件：{exc}") from exc
    details = "；".join(errors)
    raise CsvValidationError(
        f"无法识别 CSV 编码。已尝试 utf-8-sig、utf-8 和 gb18030。{details}"
    )


def validate_and_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required fields and safely clean values without losing row IDs."""
    prepared = df.copy()
    prepared.columns = [str(column).strip() for column in prepared.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in prepared.columns]
    if missing:
        raise CsvValidationError(
            "CSV 缺少必要字段。",
            missing_columns=missing,
            detected_columns=list(prepared.columns),
        )

    for column in TEXT_COLUMNS:
        prepared[column] = prepared[column].fillna("").astype(str).str.strip()
    for column in NUMERIC_COLUMNS:
        prepared[column] = (
            pd.to_numeric(prepared[column], errors="coerce").fillna(0).astype(int)
        )
    prepared = prepared.loc[prepared["单词"] != ""].copy()
    return prepared


def normalize_answer(value: str) -> str:
    return str(value).strip().casefold()


def extract_part_of_speech(meaning: str) -> tuple[str, str]:
    """Extract a leading POS marker and return (marker, cleaned meaning)."""
    original = str(meaning).strip()
    match = PART_OF_SPEECH_PATTERN.match(original)
    if not match:
        return "", original
    return match.group(1), original[match.end() :].strip()


def build_word_hint(word: str) -> str:
    """Show endpoints while preserving internal punctuation that must be typed."""
    value = str(word)
    if len(value) <= 2:
        return value
    middle = [char if char in {"-", "'", "’"} else "_" for char in value[1:-1]]
    return " ".join([value[0], *middle, value[-1]])


def build_audio_url(word: str) -> str:
    return f"https://dict.youdao.com/dictvoice?audio={quote(str(word), safe='')}&type=2"


def build_question_order(
    df: pd.DataFrame, rng: random.Random | None = None
) -> list[Any]:
    """Shuffle wrong-before-clean groups while keeping only DataFrame indexes."""
    generator = rng or random.Random()
    hard = list(df.index[df["当天错误"] > 0])
    normal = list(df.index[df["当天错误"] == 0])
    generator.shuffle(hard)
    generator.shuffle(normal)
    return hard + normal


def update_answer_counters(
    df: pd.DataFrame, row_index: Any, is_correct: bool
) -> None:
    """Update exactly one row, including when words are duplicated."""
    df.at[row_index, "当天答题次数"] = int(df.at[row_index, "当天答题次数"]) + 1
    target = "当天正确" if is_correct else "当天错误"
    df.at[row_index, target] = int(df.at[row_index, target]) + 1


def build_letter_diff_cells(
    correct_answer: str, user_answer: str
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return aligned (character, status) cells for target and input rows."""
    target = str(correct_answer)
    attempt = str(user_answer)
    matcher = difflib.SequenceMatcher(a=target.casefold(), b=attempt.casefold())
    correct_cells: list[tuple[str, str]] = []
    user_cells: list[tuple[str, str]] = []

    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        target_chunk = target[i1:i2]
        attempt_chunk = attempt[j1:j2]
        if opcode == "equal":
            for correct_char, user_char in zip(target_chunk, attempt_chunk):
                correct_cells.append((correct_char, "ok"))
                user_cells.append((user_char, "ok"))
        elif opcode == "delete":
            for correct_char in target_chunk:
                correct_cells.append((correct_char, "bad"))
                user_cells.append(("_", "missing"))
        elif opcode == "insert":
            for user_char in attempt_chunk:
                correct_cells.append(("\u00a0", "blank"))
                user_cells.append((user_char, "extra"))
        else:
            width = max(len(target_chunk), len(attempt_chunk))
            for offset in range(width):
                if offset < len(target_chunk):
                    correct_cells.append((target_chunk[offset], "bad"))
                else:
                    correct_cells.append(("\u00a0", "blank"))
                if offset < len(attempt_chunk):
                    user_cells.append((attempt_chunk[offset], "bad"))
                else:
                    user_cells.append(("_", "missing"))
    return correct_cells, user_cells


def _render_diff_cells(cells: Sequence[tuple[str, str]]) -> str:
    spans = []
    for character, status in cells:
        safe_character = html.escape(character)
        spans.append(
            f'<span class="xb-char xb-char-{status}">{safe_character}</span>'
        )
    return "".join(spans)


def build_letter_diff_html(correct_answer: str, user_answer: str) -> str:
    correct_cells, user_cells = build_letter_diff_cells(correct_answer, user_answer)
    return (
        '<div class="xb-diff" aria-label="逐字母核对">'
        '<div class="xb-diff-row"><span class="xb-diff-label">正确答案</span>'
        f"{_render_diff_cells(correct_cells)}</div>"
        '<div class="xb-diff-row"><span class="xb-diff-label">你的输入</span>'
        f"{_render_diff_cells(user_cells)}</div></div>"
    )


def save_dataframe_safely(df: pd.DataFrame, csv_path: Path) -> None:
    """Write in the same directory and atomically replace the selected CSV."""
    csv_path = csv_path.resolve()
    if not csv_path.parent.exists():
        raise OSError(f"目标文件夹不存在：{csv_path.parent}")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            suffix=".tmp.csv",
            prefix=f".{csv_path.stem}_",
            dir=csv_path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            df.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, csv_path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def create_backup_once(csv_path: Path) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = BACKUP_DIR / f"{csv_path.stem}_{timestamp}{csv_path.suffix}"
    shutil.copy2(csv_path, backup_path)
    return backup_path


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def get_supabase_storage() -> SupabaseStorage | None:
    """Build a server-side cloud client without displaying either secret."""
    try:
        config = SupabaseConfig(
            url=str(st.secrets.get("SUPABASE_URL", "")),
            service_role_key=str(
                st.secrets.get("SUPABASE_SECRET_KEY", "")
                or st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
            ),
        )
    except Exception:
        return None
    if not config.configured:
        return None
    try:
        return SupabaseStorage(config)
    except ValueError:
        return None


def current_file_record() -> dict[str, Any] | None:
    current = str(st.session_state.loaded_source_path or "")
    for record in st.session_state.file_records or []:
        if str(record.get("path") or "") == current:
            return record
    return None


def current_source_name() -> str:
    record = current_file_record()
    if record:
        return str(record.get("name") or "小白拼写.csv")
    current = str(st.session_state.loaded_source_path or "")
    if is_cloud_ref(current):
        return str(st.session_state.cloud_source_name or "小白拼写.csv")
    return Path(current).name if current else "小白拼写.csv"


def initialize_state_defaults() -> None:
    defaults: dict[str, Any] = {
        "file_records": None,
        "df": None,
        "source_path": None,
        "loaded_source_path": None,
        "source_signature": None,
        "loaded_encoding": None,
        "question_order": [],
        "current_position": 0,
        "round_id": 0,
        "answered": False,
        "last_user_answer": "",
        "last_is_correct": None,
        "last_diff_html": "",
        "recent_feedback": None,
        "session_correct": 0,
        "session_wrong": 0,
        "session_answered": 0,
        "session_wrong_records": [],
        "finished": False,
        "balloons_shown": False,
        "last_autoplay_question_key": None,
        "backed_up_paths": set(),
        "backup_paths": {},
        "save_status": "尚未修改",
        "save_error": "",
        "backup_warning": "",
        "file_error": "",
        "file_error_missing_columns": [],
        "file_error_detected_columns": [],
        "submission_error": "",
        "upload_message": "",
        "upload_error": "",
        "file_missing": False,
        "learning_cards": {},
        "learning_card_errors": {},
        "learning_cloud_checked": set(),
        "cloud_revision": None,
        "cloud_source_name": "",
        "cloud_error": "",
        "cloud_last_sync": "",
        "pending_selected_file_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_round_feedback() -> None:
    st.session_state.answered = False
    st.session_state.last_user_answer = ""
    st.session_state.last_is_correct = None
    st.session_state.last_diff_html = ""
    st.session_state.submission_error = ""
    st.session_state.backup_warning = ""


def reset_round(question_order: Sequence[Any] | None = None) -> None:
    df = st.session_state.df
    if question_order is None:
        question_order = build_question_order(df) if df is not None else []
    st.session_state.question_order = list(question_order)
    st.session_state.current_position = 0
    st.session_state.round_id += 1
    st.session_state.session_correct = 0
    st.session_state.session_wrong = 0
    st.session_state.session_answered = 0
    st.session_state.session_wrong_records = []
    st.session_state.finished = len(st.session_state.question_order) == 0
    st.session_state.balloons_shown = False
    st.session_state.last_autoplay_question_key = None
    st.session_state.recent_feedback = None
    clear_round_feedback()


def initialize_file_session(source: str | Path) -> bool:
    """Load one local or cloud word list, then create a fresh question queue."""
    source_ref = str(source)
    if not is_cloud_ref(source_ref):
        source_ref = str(Path(source_ref).resolve())
    st.session_state.source_path = source_ref
    st.session_state.file_error = ""
    st.session_state.file_error_missing_columns = []
    st.session_state.file_error_detected_columns = []

    if is_cloud_ref(source_ref):
        storage = get_supabase_storage()
        if storage is None:
            st.session_state.file_error = "Supabase 配置不完整，无法读取云端词表。"
            st.session_state.file_missing = True
            return False
        try:
            cloud_item = storage.load_word_list(cloud_id_from_ref(source_ref))
            raw_df = cloud_item["dataframe"]
            encoding = str(cloud_item.get("encoding") or "utf-8-sig")
            prepared_df = validate_and_prepare_dataframe(raw_df)
        except (CloudStorageError, CsvValidationError, ValueError) as exc:
            st.session_state.df = None
            st.session_state.loaded_source_path = source_ref
            st.session_state.file_error = str(exc)
            st.session_state.file_error_missing_columns = getattr(
                exc, "missing_columns", []
            )
            st.session_state.file_error_detected_columns = getattr(
                exc, "detected_columns", []
            )
            st.session_state.file_missing = isinstance(exc, CloudStorageError)
            st.session_state.save_status = "同步失败"
            return False
        st.session_state.df = prepared_df
        st.session_state.loaded_source_path = source_ref
        st.session_state.source_signature = None
        st.session_state.loaded_encoding = encoding
        st.session_state.cloud_revision = int(cloud_item["revision"])
        st.session_state.cloud_source_name = str(cloud_item["source_name"])
        st.session_state.cloud_last_sync = datetime.now().strftime("%H:%M:%S")
        st.session_state.file_missing = False
        st.session_state.save_status = "云端已同步"
        st.session_state.save_error = ""
        reset_round()
        return True

    resolved = Path(source_ref)
    st.session_state.file_missing = not resolved.exists()
    if not resolved.exists():
        if (
            st.session_state.loaded_source_path == source_ref
            and st.session_state.df is not None
        ):
            return False
        st.session_state.file_error = "所选 CSV 已不存在。请刷新列表或重新上传。"
        return False
    try:
        raw_df, encoding = read_csv_safely(resolved)
        prepared_df = validate_and_prepare_dataframe(raw_df)
    except CsvValidationError as exc:
        st.session_state.df = None
        st.session_state.loaded_source_path = str(resolved)
        st.session_state.file_error = str(exc)
        st.session_state.file_error_missing_columns = exc.missing_columns
        st.session_state.file_error_detected_columns = exc.detected_columns
        st.session_state.save_status = "尚未修改"
        return False

    st.session_state.df = prepared_df
    st.session_state.loaded_source_path = source_ref
    st.session_state.source_signature = file_signature(resolved)
    st.session_state.loaded_encoding = encoding
    st.session_state.cloud_revision = None
    st.session_state.cloud_source_name = ""
    st.session_state.save_status = "尚未修改"
    st.session_state.save_error = ""
    reset_round()
    return True


def refresh_file_records() -> None:
    current = str(st.session_state.loaded_source_path or "")
    records = discover_csv_files(BASE_DIR)
    storage = get_supabase_storage()
    if storage is not None:
        try:
            records.extend(storage.list_word_lists())
        except CloudStorageError as exc:
            st.session_state.cloud_error = str(exc)
        else:
            st.session_state.cloud_error = ""
    known_paths = {record["path"] for record in records}
    if current:
        st.session_state.file_missing = (
            current not in known_paths and not (
                is_cloud_ref(current) and st.session_state.cloud_error
            )
        )
    if current and current not in known_paths and st.session_state.df is not None:
        if is_cloud_ref(current):
            name = str(st.session_state.cloud_source_name or "云端词表.csv")
            records.append(
                {
                    "path": current,
                    "name": name,
                    "parent": "Supabase 云端",
                    "size": 0,
                    "mtime": datetime.now().timestamp(),
                    "date": parse_date_from_filename(name),
                    "missing": not bool(st.session_state.cloud_error),
                    "cloud": True,
                    "cloud_id": cloud_id_from_ref(current),
                    "revision": int(st.session_state.cloud_revision or 1),
                    "row_count": len(st.session_state.df),
                }
            )
        else:
            path = Path(current)
            old_signature = st.session_state.source_signature
            records.append(
                {
                    "path": current,
                    "name": path.name,
                    "parent": path.parent.name,
                    "size": old_signature[1] if old_signature else 0,
                    "mtime": (
                        old_signature[2] / 1_000_000_000
                        if old_signature
                        else datetime.now().timestamp()
                    ),
                    "date": parse_date_from_filename(path.name),
                    "missing": True,
                }
            )
    st.session_state.file_records = sort_file_records(records)


def save_uploaded_file(uploaded_file: Any) -> Path:
    """Validate uploaded bytes in a temporary file before replacing a local copy."""
    safe_name = Path(uploaded_file.name).name
    if not safe_name.casefold().endswith(".csv"):
        raise CsvValidationError("只支持 CSV 文件。")
    if safe_name in {"", ".", ".."}:
        raise CsvValidationError("上传文件名无效。")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target_path = (UPLOAD_DIR / safe_name).resolve()
    if target_path.parent != UPLOAD_DIR.resolve():
        raise CsvValidationError("上传文件名不安全。")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".upload_",
            suffix=".tmp.csv",
            dir=UPLOAD_DIR,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(uploaded_file.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        raw_df, _ = read_csv_safely(temp_path)
        validate_and_prepare_dataframe(raw_df)
        os.replace(temp_path, target_path)
        return target_path
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def handle_upload() -> None:
    uploaded_file = st.session_state.get("uploaded_csv_widget")
    if uploaded_file is None:
        return
    st.session_state.upload_error = ""
    st.session_state.upload_message = ""
    try:
        target = save_uploaded_file(uploaded_file)
        storage = get_supabase_storage()
        if storage is not None:
            raw_df, encoding = read_csv_safely(target)
            prepared_df = validate_and_prepare_dataframe(raw_df)
            row = storage.upsert_word_list(
                source_name=target.name,
                dataframe=prepared_df,
                encoding=encoding,
                source_date=parse_date_from_filename(target.name),
            )
            target_ref = cloud_ref(str(row["id"]))
            refresh_file_records()
            initialize_file_session(target_ref)
            st.session_state.selected_file_path = target_ref
            st.session_state.upload_message = (
                f"{target.name} 已上传云端，可在其他设备继续学习。"
            )
            return
        refresh_file_records()
        initialize_file_session(target)
        st.session_state.selected_file_path = str(target)
        st.session_state.upload_message = f"已保存到 {target}"
    except (CloudStorageError, CsvValidationError, OSError) as exc:
        st.session_state.upload_error = f"上传失败：{exc}"


def process_submission(user_answer: str) -> bool:
    """Persist one valid answer. Return True only after an atomic save succeeds."""
    st.session_state.submission_error = ""
    if st.session_state.answered:
        return False
    if not normalize_answer(user_answer):
        st.session_state.submission_error = "请输入单词后再提交。"
        return False

    order = st.session_state.question_order
    position = st.session_state.current_position
    if position < 0 or position >= len(order):
        st.session_state.submission_error = "当前题目状态已失效，请重新开始本轮。"
        return False

    row_index = order[position]
    df = st.session_state.df
    correct_answer = str(df.at[row_index, "单词"])
    is_correct = normalize_answer(user_answer) == normalize_answer(correct_answer)
    source_ref = str(st.session_state.loaded_source_path or "")
    cloud_mode = is_cloud_ref(source_ref)
    source_path = Path(source_ref) if not cloud_mode else None
    cloud_storage = get_supabase_storage() if cloud_mode else None
    if cloud_mode and cloud_storage is None:
        st.session_state.save_status = "同步失败"
        st.session_state.save_error = "Supabase 配置不完整。"
        st.session_state.submission_error = (
            "本题未计分：当前无法连接云端，请检查 Supabase 配置。"
        )
        return False
    if not cloud_mode and (source_path is None or not source_path.exists()):
        st.session_state.file_missing = True
        st.session_state.save_status = "保存失败"
        st.session_state.save_error = "当前 CSV 已被外部删除。"
        st.session_state.submission_error = (
            "本题未计分：当前 CSV 已不存在。请先下载内存数据或恢复文件。"
        )
        return False

    if (
        not cloud_mode
        and source_path is not None
        and str(source_path) not in st.session_state.backed_up_paths
    ):
        try:
            backup_path = create_backup_once(source_path)
            st.session_state.backed_up_paths.add(str(source_path))
            st.session_state.backup_paths[str(source_path)] = str(backup_path)
        except OSError as exc:
            st.session_state.backup_warning = (
                f"原文件备份失败，但仍会尝试保存本题：{exc}"
            )

    previous_values = {
        "当天答题次数": int(df.at[row_index, "当天答题次数"]),
        "当天正确": int(df.at[row_index, "当天正确"]),
        "当天错误": int(df.at[row_index, "当天错误"]),
    }
    previous_correct = st.session_state.session_correct
    previous_wrong = st.session_state.session_wrong
    previous_answered = st.session_state.session_answered
    previous_wrong_count = len(st.session_state.session_wrong_records)

    update_answer_counters(df, row_index, is_correct)
    st.session_state.session_answered += 1
    if is_correct:
        st.session_state.session_correct += 1
    else:
        st.session_state.session_wrong += 1
        st.session_state.session_wrong_records.append(
            {
                "row_index": row_index,
                "单词": correct_answer,
                "中文释义": str(df.at[row_index, "中文释义"]),
                "你的输入": user_answer,
                "正确答案": correct_answer,
                "类型": str(df.at[row_index, "类型"]),
            }
        )

    try:
        if cloud_mode:
            st.session_state.cloud_revision = cloud_storage.save_word_list(
                record_id=cloud_id_from_ref(source_ref),
                dataframe=df,
                expected_revision=int(st.session_state.cloud_revision or 1),
            )
            st.session_state.cloud_last_sync = datetime.now().strftime("%H:%M:%S")
        else:
            save_dataframe_safely(df, source_path)
    except (CloudStorageError, OSError) as exc:
        for column, value in previous_values.items():
            df.at[row_index, column] = value
        st.session_state.session_correct = previous_correct
        st.session_state.session_wrong = previous_wrong
        st.session_state.session_answered = previous_answered
        del st.session_state.session_wrong_records[previous_wrong_count:]
        st.session_state.save_status = "同步失败" if cloud_mode else "保存失败"
        st.session_state.save_error = str(exc)
        st.session_state.submission_error = (
            f"本题未计分：{'同步云端' if cloud_mode else '保存当前 CSV'}失败。{exc}"
        )
        return False

    if not cloud_mode and source_path is not None:
        st.session_state.source_signature = file_signature(source_path)
    st.session_state.file_missing = False
    st.session_state.save_status = "云端已同步" if cloud_mode else "已保存"
    st.session_state.save_error = ""
    st.session_state.answered = True
    st.session_state.last_user_answer = user_answer
    st.session_state.last_is_correct = is_correct
    st.session_state.last_diff_html = build_letter_diff_html(
        correct_answer, user_answer
    )
    return True


def go_to_next_question() -> None:
    if not st.session_state.answered:
        return
    next_position = st.session_state.current_position + 1
    if next_position >= len(st.session_state.question_order):
        st.session_state.finished = True
    else:
        st.session_state.current_position = next_position
        clear_round_feedback()


def remember_recent_feedback() -> None:
    """Keep a compact result visible after immediately advancing to the next word."""
    order = st.session_state.question_order
    position = st.session_state.current_position
    if position < 0 or position >= len(order):
        return
    row_index = order[position]
    st.session_state.recent_feedback = {
        "correct_answer": str(st.session_state.df.at[row_index, "单词"]),
        "user_answer": st.session_state.last_user_answer,
        "is_correct": bool(st.session_state.last_is_correct),
        "diff_html": st.session_state.last_diff_html,
        "backup_warning": st.session_state.backup_warning,
    }


def render_input_autofocus(question_key: str) -> None:
    """Focus the active spelling field using fixed, trusted local JavaScript."""
    st.html(
        f"""
        <script>
        (() => {{
            const questionKey = {question_key!r};
            const focusSpellingInput = () => {{
                const input = [...document.querySelectorAll('input')]
                    .find((element) =>
                        element.getAttribute('aria-label') === '输入完整英文单词'
                        && !element.disabled
                    );
                if (input && document.activeElement !== input) {{
                    input.focus({{ preventScroll: true }});
                    input.dataset.questionKey = questionKey;
                }}
            }};
            requestAnimationFrame(() => {{
                focusSpellingInput();
                window.setTimeout(focusSpellingInput, 80);
                window.setTimeout(focusSpellingInput, 220);
            }});
        }})();
        </script>
        """,
        width="content",
        unsafe_allow_javascript=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="xb-brand">
            <div class="xb-brand-row">
                <div class="xb-brand-identity">
                    <div class="xb-logo" aria-hidden="true">拼</div>
                    <div>
                        <h1>小白拼写</h1>
                        <p>每天一点，把单词练进手指里</p>
                    </div>
                </div>
                <div class="xb-keyboard-ready">
                    <span class="xb-keyboard-dot"></span>
                    专注模式 · 回车自动下一词
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_progress() -> None:
    total = len(st.session_state.question_order)
    answered = min(st.session_state.session_answered, total)
    correct = st.session_state.session_correct
    accuracy = correct / answered if answered else 0.0
    remaining = max(total - answered, 0)
    percent = min(max(answered / total if total else 0.0, 0.0), 1.0)
    st.markdown(
        f"""
        <div class="xb-session-rail">
            <div class="xb-session-count">
                <strong>{answered}</strong><span>/ {total}</span>
            </div>
            <div class="xb-session-track" role="progressbar"
                 aria-valuenow="{answered}" aria-valuemin="0" aria-valuemax="{total}">
                <span style="width:{percent * 100:.2f}%"></span>
            </div>
            <div class="xb-session-metrics">
                <div class="xb-metric"><span>正确</span><strong>{correct}</strong></div>
                <div class="xb-metric"><span>正确率</span><strong>{accuracy:.0%}</strong></div>
                <div class="xb-metric"><span>剩余</span><strong>{remaining}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recent_feedback() -> None:
    feedback = st.session_state.recent_feedback
    if not feedback:
        return
    safe_correct = html.escape(str(feedback["correct_answer"]))
    safe_user = html.escape(str(feedback["user_answer"]))
    safe_warning = html.escape(str(feedback.get("backup_warning", "")))
    warning_html = (
        f'<div class="xb-recent-warning">{safe_warning}</div>'
        if safe_warning
        else ""
    )
    if feedback["is_correct"]:
        st.markdown(
            f"""
            <div class="xb-recent xb-recent-ok">
                <div class="xb-recent-main">
                    <div class="xb-recent-title"><span>✓</span> 上一词拼写正确</div>
                    <div class="xb-recent-answer">{safe_correct}</div>
                </div>
                {warning_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f"""
        <div class="xb-recent xb-recent-bad">
            <div class="xb-recent-main">
                <div class="xb-recent-title"><span>!</span> 上一词需要再留意</div>
                <div class="xb-recent-answer">{safe_user} → {safe_correct}</div>
            </div>
            {warning_html}
            {feedback["diff_html"]}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer_feedback() -> None:
    if st.session_state.submission_error:
        st.warning(st.session_state.submission_error)
    if st.session_state.backup_warning:
        st.warning(st.session_state.backup_warning)
    if not st.session_state.answered:
        return

    row_index = st.session_state.question_order[st.session_state.current_position]
    correct_answer = str(st.session_state.df.at[row_index, "单词"])
    user_answer = st.session_state.last_user_answer
    safe_correct = html.escape(correct_answer)
    safe_user = html.escape(user_answer)
    if st.session_state.last_is_correct:
        st.markdown(
            '<div class="xb-feedback xb-feedback-ok">拼写正确。手指已经记住它了。</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="xb-feedback xb-feedback-bad">再看一眼字母顺序，下次会更稳。</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f"""
        <div class="xb-answer-pair">
            正确答案&nbsp; <code>{safe_correct}</code><br>
            你的输入&nbsp; <code>{safe_user or "（空）"}</code>
        </div>
        {st.session_state.last_diff_html}
        """,
        unsafe_allow_html=True,
    )


def get_nvidia_settings() -> tuple[str, tuple[str, ...]]:
    """Read optional secrets without ever logging or displaying the API key."""
    try:
        api_key = str(st.secrets.get("NVIDIA_API_KEY", "")).strip()
        primary = str(st.secrets.get("NVIDIA_MODEL", "")).strip()
        fallback = str(
            st.secrets.get("NVIDIA_FALLBACK_MODEL", "")
        ).strip()
    except Exception:
        return "", ()
    models = tuple(
        dict.fromkeys(model for model in (primary, fallback) if model)
    )
    return api_key, models


def render_learning_bundle(bundle: dict[str, Any]) -> None:
    dictionary = bundle.get("dictionary") or {}
    ai_card = bundle.get("ai") or {}
    safe_model = html.escape(str(bundle.get("model") or "NVIDIA NIM"))
    safe_example_en = html.escape(str(ai_card.get("example_en") or ""))
    safe_example_zh = html.escape(str(ai_card.get("example_zh") or ""))
    safe_usage = html.escape(str(ai_card.get("usage_note") or ""))
    safe_spelling = html.escape(str(ai_card.get("spelling_tip") or ""))
    safe_mnemonic = html.escape(str(ai_card.get("mnemonic") or ""))
    family = [
        html.escape(str(item))
        for item in (ai_card.get("word_family") or [])
        if str(item).strip()
    ]
    family_html = (
        '<div class="xb-family">'
        + "".join(f"<span>{item}</span>" for item in family)
        + "</div>"
        if family
        else ""
    )

    origin = str(dictionary.get("origin") or "").strip()
    definition_en = str(dictionary.get("definition_en") or "").strip()
    phonetic = str(dictionary.get("phonetic") or "").strip()
    dictionary_error = str(bundle.get("dictionary_error") or "").strip()
    if origin:
        origin_text = (
            "<strong>已核验词源资料 · DictionaryAPI</strong><br>"
            f"{html.escape(origin)}"
        )
    elif dictionary_error:
        origin_text = (
            "<strong>词典查询暂时失败</strong><br>"
            f"{html.escape(dictionary_error)} "
            "上面的“联想”只是记忆提示，不代表真实词源。"
        )
    else:
        origin_text = (
            "<strong>词源说明</strong><br>"
            "DictionaryAPI 未返回可核验词源；上面的“联想”只是记忆提示，"
            "不会冒充真实词根或词源。"
        )
    dictionary_bits = " · ".join(
        bit
        for bit in (
            html.escape(phonetic),
            html.escape(definition_en),
        )
        if bit
    )
    dictionary_html = (
        f'<div class="xb-ai-origin">{dictionary_bits}<br>{origin_text}</div>'
        if dictionary_bits
        else f'<div class="xb-ai-origin">{origin_text}</div>'
    )

    st.markdown(
        f"""
        <div class="xb-ai-shell">
            <div class="xb-ai-head">
                <div>
                    <div class="xb-ai-title">
                        <span class="xb-ai-spark">✦</span>
                        本词记忆卡
                    </div>
                    <div class="xb-ai-subtitle">
                        词典事实与 AI 联想分开显示 · 已保存到安全缓存
                    </div>
                </div>
                <div class="xb-ai-model" title="{safe_model}">{safe_model}</div>
            </div>
            <div class="xb-ai-grid">
                <div class="xb-ai-item xb-ai-item-wide">
                    <div class="xb-ai-label">Example</div>
                    <div class="xb-ai-example">{safe_example_en}</div>
                    <div class="xb-ai-translation">{safe_example_zh}</div>
                </div>
                <div class="xb-ai-item">
                    <div class="xb-ai-label">Usage</div>
                    <div class="xb-ai-value">{safe_usage}</div>
                </div>
                <div class="xb-ai-item">
                    <div class="xb-ai-label">Spelling</div>
                    <div class="xb-ai-value">{safe_spelling}</div>
                </div>
                <div class="xb-ai-item xb-ai-item-wide">
                    <div class="xb-ai-label">Mnemonic</div>
                    <div class="xb-ai-value">{safe_mnemonic}</div>
                    {family_html}
                </div>
            </div>
            {dictionary_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_learning_assistant(
    word: str,
    chinese_meaning: str,
    question_key: str,
) -> None:
    normalized_word = normalize_answer(word)
    cards = st.session_state.learning_cards
    errors = st.session_state.learning_card_errors
    cloud_checked = st.session_state.learning_cloud_checked
    cloud_storage = get_supabase_storage()
    if (
        normalized_word not in cards
        and cloud_storage is not None
        and normalized_word not in cloud_checked
    ):
        cloud_checked.add(normalized_word)
        try:
            cloud_bundle = cloud_storage.load_ai_card(word)
        except CloudStorageError:
            cloud_bundle = None
        if cloud_bundle is not None:
            cards[normalized_word] = cloud_bundle
    if normalized_word not in cards:
        cached = load_cached_bundle(LEARNING_CACHE_DIR, word)
        if cached is not None:
            cards[normalized_word] = cached
    bundle = cards.get(normalized_word)
    api_key, models = get_nvidia_settings()

    with st.expander(
        "✦ AI 记忆助手 · 例句 / 拼写 / 联想",
        expanded=bundle is not None,
    ):
        if bundle is not None:
            render_learning_bundle(bundle)
        else:
            st.markdown(
                """
                <div class="xb-ai-idle">
                    点击后才会查询词典并调用 NVIDIA。不会在每次按回车时自动请求，
                    因此不会拖慢连续拼写；生成成功后会保存到本地缓存。
                </div>
                """,
                unsafe_allow_html=True,
            )

        if not api_key or not models:
            st.warning(
                "AI 配置尚未完成。请在 .streamlit/secrets.toml 中设置密钥和模型。"
            )
            return

        button_label = "重新生成本词记忆卡" if bundle else "生成本词记忆卡"
        if st.button(
            button_label,
            key=f"learning-card-{question_key}",
            type="secondary",
            width="stretch",
        ):
            errors.pop(normalized_word, None)
            with st.spinner("正在查询词典并生成记忆卡…"):
                try:
                    generated = generate_learning_bundle(
                        word=word,
                        chinese_meaning=chinese_meaning,
                        api_key=api_key,
                        models=models,
                        cache_dir=LEARNING_CACHE_DIR,
                    )
                except LearningAssistantError as exc:
                    errors[normalized_word] = str(exc)
                else:
                    cards[normalized_word] = generated
                    if cloud_storage is not None:
                        try:
                            cloud_storage.save_ai_card(word, generated)
                        except CloudStorageError as exc:
                            st.warning(
                                f"记忆卡已生成，但云端缓存同步失败：{exc}"
                            )
                        else:
                            st.success("记忆卡已生成并同步到云端。")
                    else:
                        st.success("记忆卡已生成并保存到本地。")
                    render_learning_bundle(generated)

        if errors.get(normalized_word):
            st.error(errors[normalized_word])


def render_question_card() -> None:
    order = st.session_state.question_order
    position = st.session_state.current_position
    if not order or position >= len(order):
        st.session_state.finished = True
        st.rerun()
    row_index = order[position]
    row = st.session_state.df.loc[row_index]
    word = str(row["单词"])
    learning_type = str(row["类型"]) or "未分类"
    pos, cleaned_meaning = extract_part_of_speech(str(row["中文释义"]))
    display_meaning = cleaned_meaning if pos else str(row["中文释义"])
    if not display_meaning:
        display_meaning = "暂无释义"
    mode = st.session_state.practice_mode
    mode_letter = mode.split("·", maxsplit=1)[0].strip()

    safe_type = html.escape(learning_type)
    safe_meaning = html.escape(display_meaning)
    safe_pos = html.escape(pos)
    safe_word = html.escape(word)
    if mode.startswith("模式 A"):
        focal_html = f'<div class="xb-word">{safe_word}</div>'
    elif mode.startswith("模式 B"):
        focal_html = (
            f'<div class="xb-hint">{html.escape(build_word_hint(word))}</div>'
        )
    else:
        focal_html = '<div class="xb-hidden-word">听发音 · 看释义 · 完整拼写</div>'

    render_recent_feedback()
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="xb-card-head">
                <div class="xb-card-labels">
                    <span class="xb-pill">{html.escape(mode_letter)}</span>
                    <span class="xb-pill xb-pill-neutral">{safe_type}</span>
                </div>
                <span class="xb-card-number">WORD {position + 1:02d}</span>
            </div>
            <div class="xb-prompt-zone">
                {focal_html}
                <div class="xb-meaning">{safe_meaning}</div>
                {f'<div class="xb-pos">{safe_pos}</div>' if safe_pos else ''}
            </div>
            <div class="xb-audio-label">美式发音 · 点击可重播</div>
            """,
            unsafe_allow_html=True,
        )

        question_key = (
            f"{hashlib.sha1(str(st.session_state.loaded_source_path).encode()).hexdigest()[:10]}"
            f"-{st.session_state.round_id}-{position}-{row_index}"
        )
        should_autoplay = (
            st.session_state.last_autoplay_question_key != question_key
        )
        if should_autoplay:
            st.session_state.last_autoplay_question_key = question_key
        try:
            st.audio(
                build_audio_url(word),
                format="audio/mpeg",
                autoplay=should_autoplay,
            )
        except Exception:
            st.caption("发音暂时不可用，不影响继续答题。")

        st.markdown(
            """
            <div class="xb-input-guide">
                <strong>键盘已经准备好</strong>
                <span>·</span>
                <kbd>Enter ↵</kbd>
                <span>提交并自动进入下一词</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        input_key = f"answer-{question_key}"
        with st.form(key=f"answer-form-{question_key}", clear_on_submit=False):
            user_answer = st.text_input(
                "输入完整英文单词",
                placeholder="输入完整单词，然后按 Enter",
                disabled=st.session_state.answered,
                key=input_key,
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button(
                "提交并进入下一词  ↵",
                type="primary",
                width="stretch",
            )
        render_input_autofocus(question_key)
        if submitted:
            if process_submission(user_answer):
                remember_recent_feedback()
                go_to_next_question()
            st.rerun()

        render_answer_feedback()

    render_learning_assistant(word, display_meaning, question_key)


def render_completion_page() -> None:
    if not st.session_state.balloons_shown:
        st.balloons()
        st.session_state.balloons_shown = True

    total = st.session_state.session_answered
    correct = st.session_state.session_correct
    wrong = st.session_state.session_wrong
    accuracy = correct / total if total else 0.0
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="xb-complete">
                <div class="xb-complete-mark">✓</div>
                <h2>今天的字母训练完成</h2>
                <p>你已经把这一组单词完整地敲过一遍。</p>
            </div>
            <div class="xb-stat-grid">
                <div class="xb-stat">
                    <div class="xb-stat-label">本轮题数</div>
                    <div class="xb-stat-value">{total}</div>
                </div>
                <div class="xb-stat">
                    <div class="xb-stat-label">正确</div>
                    <div class="xb-stat-value">{correct}</div>
                </div>
                <div class="xb-stat">
                    <div class="xb-stat-label">错误</div>
                    <div class="xb-stat-value">{wrong}</div>
                </div>
                <div class="xb-stat">
                    <div class="xb-stat-label">正确率</div>
                    <div class="xb-stat-value">{accuracy:.0%}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("本轮回顾")
        wrong_records = st.session_state.session_wrong_records
        if wrong_records:
            display_records = [
                {key: record[key] for key in ("单词", "中文释义", "你的输入", "正确答案", "类型")}
                for record in wrong_records
            ]
            st.dataframe(
                pd.DataFrame(display_records),
                hide_index=True,
                width="stretch",
            )
        else:
            st.success("这一轮全部拼写正确，很稳。")

        first, second = st.columns(2)
        with first:
            if st.button("再来一轮", type="primary", width="stretch"):
                reset_round()
                st.rerun()
        with second:
            if wrong_records and st.button(
                "只练本轮错题", width="stretch"
            ):
                unique_indexes = list(
                    dict.fromkeys(record["row_index"] for record in wrong_records)
                )
                random.shuffle(unique_indexes)
                reset_round(unique_indexes)
                st.rerun()

        filename = current_source_name()
        st.download_button(
            "下载当前 CSV",
            data=dataframe_to_csv_bytes(st.session_state.df),
            file_name=filename,
            mime="text/csv",
            width="stretch",
        )


def render_empty_state() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with st.container(border=True):
        st.markdown(
            """
            <div class="xb-empty">
                <h2>准备好今天的词表</h2>
                <p>
                    请把每天导出的 CSV 放入 <code>vocabulary_data</code> 文件夹，<br>
                    或者通过左侧的上传入口添加。
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_file_error() -> None:
    st.error(st.session_state.file_error)
    missing = st.session_state.file_error_missing_columns
    detected = st.session_state.file_error_detected_columns
    if missing:
        st.markdown("**缺失字段：** " + "、".join(map(str, missing)))
    if detected:
        st.markdown("**当前检测到：** " + "、".join(map(str, detected)))
    st.info("请重新从背词软件导出，或修正 CSV 后点击左侧“从磁盘重新加载”。")


def render_sidebar_upload() -> None:
    st.markdown('<div class="xb-side-section">数据管理</div>', unsafe_allow_html=True)
    st.file_uploader(
        "上传 CSV",
        type=["csv"],
        key="uploaded_csv_widget",
        on_change=handle_upload,
        help=(
            "已配置 Supabase 时会上传云端；否则安全保存到 "
            "uploaded_csv 本地副本。"
        ),
    )
    if st.session_state.upload_message:
        st.success(st.session_state.upload_message)
    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)
    if st.session_state.save_error:
        st.error(f"保存失败：{st.session_state.save_error}")


def render_sidebar() -> None:
    records = st.session_state.file_records or []
    with st.sidebar:
        st.markdown(
            """
            <div class="xb-side-brand">
                <div class="xb-side-mark">XB</div>
                <div><strong>小白拼写</strong><span>每日键盘拼写训练</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="xb-side-section">今日词表</div>', unsafe_allow_html=True)

        if st.button("刷新文件列表", width="stretch"):
            refresh_file_records()

        records = st.session_state.file_records or []
        if records:
            options = [record["path"] for record in records]
            record_map = {record["path"]: record for record in records}
            current = st.session_state.loaded_source_path
            pending_selected = st.session_state.pop(
                "pending_selected_file_path",
                None,
            )
            if pending_selected in options:
                st.session_state.selected_file_path = pending_selected
            if "selected_file_path" not in st.session_state:
                st.session_state.selected_file_path = (
                    current if current in options else options[0]
                )
            elif st.session_state.selected_file_path not in options:
                st.session_state.selected_file_path = (
                    current if current in options else options[0]
                )
            selected = st.selectbox(
                "选择 CSV",
                options=options,
                format_func=lambda path: build_file_label(record_map[path], records),
                key="selected_file_path",
            )
            if selected != st.session_state.loaded_source_path:
                initialize_file_session(selected)
        else:
            st.info("还没有词表，请先在下方上传 CSV。")
            render_sidebar_upload()

        if st.session_state.loaded_source_path:
            record = current_file_record() or {}
            source_ref = str(st.session_state.loaded_source_path)
            source_name = str(record.get("name") or current_source_name())
            source_parent = str(
                record.get("parent")
                or (
                    "Supabase 云端"
                    if is_cloud_ref(source_ref)
                    else Path(source_ref).parent.name
                )
            )
            row_count = len(st.session_state.df) if st.session_state.df is not None else 0
            date_label = str(
                record.get("date") or parse_date_from_filename(source_name) or ""
            )
            if not date_label:
                signature = st.session_state.source_signature
                date_label = (
                    datetime.fromtimestamp(signature[2] / 1_000_000_000).strftime(
                        "%Y-%m-%d"
                    )
                    if signature
                    else "日期未知"
                )
            status_class = {
                "已保存": "xb-save-ok",
                "云端已同步": "xb-save-ok",
                "保存失败": "xb-save-bad",
                "同步失败": "xb-save-bad",
            }.get(st.session_state.save_status, "xb-save-idle")
            st.markdown(
                f"""
                <div class="xb-file-meta">
                    <strong>{html.escape(source_name)}</strong><br>
                    位置：{html.escape(source_parent)}<br>
                    日期：{html.escape(date_label)}<br>
                    单词：{row_count} 个<br>
                    状态：<span class="{status_class}">{html.escape(st.session_state.save_status)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.session_state.file_missing:
                st.warning("当前文件已被移除；内存数据仍可下载。")

        st.markdown('<div class="xb-side-section">云端同步</div>', unsafe_allow_html=True)
        cloud_storage = get_supabase_storage()
        if cloud_storage is None:
            st.markdown(
                """
                <div class="xb-ai-status xb-ai-status-off">
                    <span class="xb-ai-status-dot"></span>
                    本地模式 · 尚未配置 Supabase
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif st.session_state.cloud_error:
            st.error(st.session_state.cloud_error)
        else:
            sync_detail = (
                f" · 版本 {int(st.session_state.cloud_revision)}"
                if is_cloud_ref(st.session_state.loaded_source_path)
                and st.session_state.cloud_revision
                else ""
            )
            st.markdown(
                f"""
                <div class="xb-ai-status">
                    <span class="xb-ai-status-dot"></span>
                    Supabase 已连接{html.escape(sync_detail)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        current_ref = str(st.session_state.loaded_source_path or "")
        can_import_local = bool(
            cloud_storage is not None
            and st.session_state.df is not None
            and current_ref
            and not is_cloud_ref(current_ref)
        )
        if st.button(
            "将当前本地词表导入云端",
            width="stretch",
            disabled=not can_import_local,
        ):
            try:
                row = cloud_storage.upsert_word_list(
                    source_name=Path(current_ref).name,
                    dataframe=st.session_state.df,
                    encoding=str(
                        st.session_state.loaded_encoding or "utf-8-sig"
                    ),
                    source_date=parse_date_from_filename(Path(current_ref).name),
                )
                target_ref = cloud_ref(str(row["id"]))
                refresh_file_records()
                st.session_state.pending_selected_file_path = target_ref
                st.session_state.upload_message = "当前词表已安全导入云端。"
                st.rerun()
            except (CloudStorageError, CsvValidationError, OSError) as exc:
                st.error(f"导入失败：{exc}")

        st.markdown('<div class="xb-side-section">练习模式</div>', unsafe_allow_html=True)
        st.radio(
            "选择模式",
            PRACTICE_MODES,
            key="practice_mode",
            label_visibility="collapsed",
        )

        st.markdown('<div class="xb-side-section">AI 记忆助手</div>', unsafe_allow_html=True)
        api_key, models = get_nvidia_settings()
        if api_key and models:
            safe_model = html.escape(models[0])
            st.markdown(
                f"""
                <div class="xb-ai-status" title="{safe_model}">
                    <span class="xb-ai-status-dot"></span>
                    已配置 · 按需生成记忆卡
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="xb-ai-status xb-ai-status-off">
                    <span class="xb-ai-status-dot"></span>
                    尚未配置 NVIDIA
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="xb-side-section">学习控制</div>', unsafe_allow_html=True)
        controls_disabled = st.session_state.df is None
        if st.button(
            "重新开始本轮",
            width="stretch",
            disabled=controls_disabled,
        ):
            reset_round()
            st.rerun()
        reload_label = (
            "从云端重新加载"
            if is_cloud_ref(st.session_state.loaded_source_path)
            else "从磁盘重新加载"
        )
        if st.button(
            reload_label,
            width="stretch",
            disabled=not st.session_state.loaded_source_path,
        ):
            initialize_file_session(st.session_state.loaded_source_path)
            st.rerun()

        if records:
            render_sidebar_upload()


def main() -> None:
    configure_page()
    inject_css()
    initialize_state_defaults()
    for directory in (DATA_DIR, UPLOAD_DIR, BACKUP_DIR, LEARNING_CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if st.session_state.file_records is None:
        refresh_file_records()

    records = st.session_state.file_records or []
    if st.session_state.loaded_source_path is None and records:
        initialize_file_session(records[0]["path"])
        st.session_state.selected_file_path = records[0]["path"]

    render_sidebar()
    render_header()

    if st.session_state.file_error:
        render_file_error()
        return
    if st.session_state.df is None:
        render_empty_state()
        return
    if st.session_state.df.empty:
        st.info("当前 CSV 中没有可练习的单词。")
        return
    if st.session_state.file_missing:
        st.error("当前词表已无法访问。你可以继续查看内存数据或下载最新副本。")
        st.download_button(
            "下载内存中的当前 CSV",
            data=dataframe_to_csv_bytes(st.session_state.df),
            file_name=current_source_name(),
            mime="text/csv",
        )

    render_progress()
    if st.session_state.finished:
        render_completion_page()
    else:
        render_question_card()


if __name__ == "__main__":
    main()
