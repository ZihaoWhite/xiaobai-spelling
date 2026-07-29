import random
from pathlib import Path

import pandas as pd
import pytest

from app import (
    CsvValidationError,
    build_audio_url,
    build_file_label,
    build_letter_diff_cells,
    build_question_order,
    build_word_hint,
    choose_initial_sidebar_state,
    extract_part_of_speech,
    normalize_answer,
    sort_file_records,
    update_answer_counters,
    validate_and_prepare_dataframe,
)


def valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "单词": ["promote", "well-being", "promote"],
            "中文释义": ["v. 促进；推动", "n. 幸福；健康", "推动"],
            "类型": ["新学", "复习", "复习"],
            "当前状态": ["1", None, "bad"],
            "当天答题次数": [3, "2", None],
            "当天正确": [2, 1, ""],
            "当天错误": [1, 0, "4"],
        },
        index=[10, 20, 30],
    )


def test_normalize_answer_ignores_case_and_outer_spaces() -> None:
    assert normalize_answer(" Promote ") == normalize_answer("promote")


def test_normalize_answer_preserves_hyphen() -> None:
    assert normalize_answer("well-being") != normalize_answer("wellbeing")


def test_audio_url_encodes_the_complete_word() -> None:
    assert build_audio_url("rock / roll") == (
        "https://dict.youdao.com/dictvoice?audio=rock%20%2F%20roll&type=2"
    )


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("a", "a"),
        ("go", "go"),
        ("apple", "a _ _ _ e"),
        ("well-being", "w _ _ _ - _ _ _ _ g"),
        ("don't", "d _ _ ' t"),
    ],
)
def test_build_word_hint(word: str, expected: str) -> None:
    assert build_word_hint(word) == expected


def test_extract_part_of_speech() -> None:
    assert extract_part_of_speech("vt. 促进；推动") == ("vt.", "促进；推动")
    assert extract_part_of_speech("adj. 稳定的") == ("adj.", "稳定的")
    assert extract_part_of_speech("新学单词") == ("", "新学单词")


def test_file_sorting_export_first_then_mtime() -> None:
    records = [
        {"path": "/a/recent.csv", "name": "recent.csv", "mtime": 30},
        {"path": "/a/export_old.csv", "name": "export_old.csv", "mtime": 10},
        {"path": "/a/export_new.csv", "name": "EXPORT_new.csv", "mtime": 20},
    ]
    sorted_names = [record["name"] for record in sort_file_records(records)]
    assert sorted_names == ["EXPORT_new.csv", "export_old.csv", "recent.csv"]


def test_sidebar_expands_when_no_csv_exists(tmp_path: Path) -> None:
    assert choose_initial_sidebar_state(tmp_path) == "expanded"
    (tmp_path / "today.csv").write_text("placeholder", encoding="utf-8")
    assert choose_initial_sidebar_state(tmp_path) == "collapsed"


def test_duplicate_filename_label_includes_parent() -> None:
    records = [
        {
            "path": "/x/vocabulary_data/export.csv",
            "name": "export.csv",
            "parent": "vocabulary_data",
            "mtime": 1,
            "date": None,
        },
        {
            "path": "/x/uploaded_csv/export.csv",
            "name": "export.csv",
            "parent": "uploaded_csv",
            "mtime": 1,
            "date": None,
        },
    ]
    assert "vocabulary_data / export.csv" in build_file_label(records[0], records)


def test_validation_reports_missing_columns() -> None:
    with pytest.raises(CsvValidationError) as error:
        validate_and_prepare_dataframe(pd.DataFrame({"单词": ["apple"]}))
    assert "中文释义" in error.value.missing_columns
    assert error.value.detected_columns == ["单词"]


def test_validation_cleans_numeric_fields_and_keeps_original_index() -> None:
    prepared = validate_and_prepare_dataframe(valid_dataframe())
    assert list(prepared.index) == [10, 20, 30]
    assert prepared.loc[20, "当前状态"] == 0
    assert prepared.loc[30, "当天答题次数"] == 0
    assert all(pd.api.types.is_integer_dtype(prepared[column]) for column in (
        "当前状态", "当天答题次数", "当天正确", "当天错误"
    ))


def test_validation_filters_empty_word_without_reindexing() -> None:
    frame = valid_dataframe()
    frame.loc[20, "单词"] = "   "
    prepared = validate_and_prepare_dataframe(frame)
    assert list(prepared.index) == [10, 30]


def test_wrong_words_are_all_before_clean_words() -> None:
    prepared = validate_and_prepare_dataframe(valid_dataframe())
    order = build_question_order(prepared, random.Random(42))
    errors = [prepared.at[index, "当天错误"] for index in order]
    first_zero = errors.index(0)
    assert all(value > 0 for value in errors[:first_zero])
    assert all(value == 0 for value in errors[first_zero:])


def test_duplicate_word_only_updates_requested_row() -> None:
    prepared = validate_and_prepare_dataframe(valid_dataframe())
    first_before = prepared.loc[10, ["当天答题次数", "当天正确"]].copy()
    third_before = prepared.loc[30, ["当天答题次数", "当天正确"]].copy()

    update_answer_counters(prepared, 30, True)

    assert prepared.at[10, "当天答题次数"] == first_before["当天答题次数"]
    assert prepared.at[10, "当天正确"] == first_before["当天正确"]
    assert prepared.at[30, "当天答题次数"] == third_before["当天答题次数"] + 1
    assert prepared.at[30, "当天正确"] == third_before["当天正确"] + 1


def test_character_diff_keeps_suffix_aligned_after_omission() -> None:
    correct_cells, user_cells = build_letter_diff_cells(
        "transcendental", "transendental"
    )
    assert len(correct_cells) == len(user_cells)
    missing_positions = [
        index for index, (_, status) in enumerate(user_cells) if status == "missing"
    ]
    assert len(missing_positions) == 1
    missing_index = missing_positions[0]
    assert correct_cells[missing_index][0] == "c"
    assert all(status == "ok" for _, status in user_cells[missing_index + 1 :])


def test_character_diff_marks_extra_character() -> None:
    correct_cells, user_cells = build_letter_diff_cells("apple", "appple")
    assert len(correct_cells) == len(user_cells)
    assert [char for char, status in user_cells if status == "extra"] == ["p"]


def test_no_standard_library_dependency_is_expected() -> None:
    requirements = (Path(__file__).parents[1] / "requirements.txt").read_text()
    assert "pathlib" not in requirements
    assert "difflib" not in requirements
