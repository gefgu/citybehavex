from __future__ import annotations

from citybehavex.tessellation import load_category_mapping
from citybehavex.tessellation.builder import purpose_distribution

import pandas as pd


def test_category_mapping_collapses_to_home_work_other():
    mapping = load_category_mapping()

    assert mapping["corporate_office"] == "WORK"
    assert mapping["cafe"] == "OTHER"
    assert mapping["college_university"] == "OTHER"
    assert mapping["hospital"] == "OTHER"


def test_purpose_distribution_collapses_cached_rich_labels():
    df = pd.DataFrame({"purpose": ["WORK", "PURCHASE", "OTHER", None]})

    distribution = purpose_distribution(df)

    assert distribution == {"OTHER": 0.75, "WORK": 0.25}
