from __future__ import annotations

from citybehavex.tessellation import load_category_mapping


def test_category_mapping_collapses_to_home_work_other():
    mapping = load_category_mapping()

    assert mapping["corporate_office"] == "WORK"
    assert mapping["cafe"] == "OTHER"
    assert mapping["college_university"] == "OTHER"
    assert mapping["hospital"] == "OTHER"
