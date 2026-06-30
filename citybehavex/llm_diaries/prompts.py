from __future__ import annotations

import json
from typing import Optional

from .models import Diary

# Cap how many prior schedules are echoed back into a prompt (token budget).
_MAX_PREVIOUS_SCHEDULES = 20


def diary_episode_summary(diary: Diary) -> str:
    """Compact one-line ``HH:MM-HH:MM PURPOSE | ...`` summary of a diary."""
    return " | ".join(f"{ep.start}-{ep.end} {ep.purpose}" for ep in diary.episodes)


def build_single_diary_prompt(
    *,
    diary_number: int,
    diary_count: int,
    city_profile: str,
    representative_day: str,
    purpose_distribution: Optional[dict[str, float]] = None,
    location_count: Optional[int] = None,
    previous_diaries: Optional[list[Diary]] = None,
) -> str:
    distribution = purpose_distribution or {}
    location_rule = ""
    if location_count == 1:
        location_rule = (
            "This is a one-location schedule: HOME is the only visited location. "
            "Return exactly one episode from 00:00 to 24:00 with purpose HOME. "
            "Do not include any non-HOME purpose.\n"
        )
    elif location_count is not None:
        location_rule = (
            f"This schedule should visit exactly {location_count} distinct "
            f"location(s) counting HOME (so {max(location_count - 1, 0)} non-home "
            "place(s)); returning to a place already visited does not add to that count.\n"
        )

    dedup_rule = ""
    if location_count != 1 and previous_diaries:
        shown = previous_diaries[-_MAX_PREVIOUS_SCHEDULES:]
        listing = "".join(
            f"  {i}. {diary_episode_summary(d)}\n" for i, d in enumerate(shown, start=1)
        )
        dedup_rule = (
            f"The following {len(shown)} schedule(s) with this same location count "
            "have already been generated. Produce a clearly different routine (vary "
            "wake/sleep and activity timing and the non-home activity mix); do NOT "
            "repeat or trivially rephrase any of them:\n"
            f"{listing}"
        )

    return (
        "Return JSON only for one synthetic daily mobility diary.\n"
        "Shape: {\"diary_id\":\"d1\",\"episodes\":[{\"start\":\"00:00\",\"end\":\"07:00\","
        "\"purpose\":\"HOME\"}]}.\n"
        f"Representative day: {representative_day}\n"
        f"Diary number: {diary_number} of {diary_count}\n"
        f"City profile: {city_profile or 'No additional city profile provided.'}\n"
        f"Purpose distribution hints: {json.dumps(distribution, sort_keys=True)}\n"
        f"{location_rule}"
        f"{dedup_rule}"
        "Allowed purposes: HOME, WORK, STUDIES, PURCHASE, LEISURE, HEALTH, OTHER.\n"
        "Rules: start at 00:00, end at 24:00, use contiguous non-overlapping episodes, "
        "include HOME, and use only start/end times in HH:MM.\n"
        "Vary this diary from the others by routine timing and non-home activity mix.\n"
        "No descriptions, notes, markdown, or commentary.\n"
    )
