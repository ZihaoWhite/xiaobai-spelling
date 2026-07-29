import json
from urllib.error import HTTPError

import pandas as pd
import pytest

from cloud_storage import (
    CloudConflictError,
    SupabaseConfig,
    SupabaseStorage,
    cloud_id_from_ref,
    cloud_ref,
    dataframe_from_payload,
    dataframe_to_payload,
    is_cloud_ref,
)


class FakeResponse:
    def __init__(self, payload):
        if payload is None:
            self.payload = b""
        else:
            self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


def test_cloud_reference_round_trip() -> None:
    value = cloud_ref("abc-123")
    assert is_cloud_ref(value)
    assert cloud_id_from_ref(value) == "abc-123"


def test_dataframe_payload_preserves_index_and_chinese_columns() -> None:
    frame = pd.DataFrame(
        {"单词": ["apple"], "当天正确": [2]},
        index=[9],
    )
    restored = dataframe_from_payload(dataframe_to_payload(frame))
    assert list(restored.index) == [9]
    assert restored.at[9, "单词"] == "apple"
    assert restored.at[9, "当天正确"] == 2


def test_list_word_lists_does_not_expose_key_in_record() -> None:
    def fake_urlopen(request, timeout, context):
        assert request.headers["Apikey"] == "server-secret"
        return FakeResponse(
            [
                {
                    "id": "list-1",
                    "source_name": "today.csv",
                    "source_date": "2026-07-29",
                    "encoding": "utf-8-sig",
                    "row_count": 3,
                    "revision": 2,
                    "updated_at": "2026-07-29T12:00:00+00:00",
                }
            ]
        )

    storage = SupabaseStorage(
        SupabaseConfig("https://example.supabase.co", "server-secret"),
        urlopen_fn=fake_urlopen,
    )
    records = storage.list_word_lists()
    assert records[0]["path"] == "cloud://list-1"
    assert records[0]["parent"] == "Supabase 云端"
    assert "server-secret" not in json.dumps(records, ensure_ascii=False)


def test_optimistic_save_reports_conflict_when_no_row_matches() -> None:
    def fake_urlopen(request, timeout, context):
        return FakeResponse([])

    storage = SupabaseStorage(
        SupabaseConfig("https://example.supabase.co", "server-secret"),
        urlopen_fn=fake_urlopen,
    )
    with pytest.raises(CloudConflictError):
        storage.save_word_list(
            record_id="list-1",
            dataframe=pd.DataFrame({"单词": ["apple"]}),
            expected_revision=2,
        )


def test_http_auth_error_is_user_friendly() -> None:
    def fake_urlopen(request, timeout, context):
        raise HTTPError(request.full_url, 401, "bad key", None, None)

    storage = SupabaseStorage(
        SupabaseConfig("https://example.supabase.co", "server-secret"),
        urlopen_fn=fake_urlopen,
    )
    with pytest.raises(Exception, match="密钥无效"):
        storage.list_word_lists()
