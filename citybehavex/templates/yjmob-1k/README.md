# CityBehavEx YJMOB-1k example

1. Run `citybehavex data download yjmob --project .`.
2. Run `citybehavex doctor --config configs/yjmob-1k.yaml`.
3. Run `citybehavex simulate --config configs/yjmob-1k.yaml --start-aligners`.

No LLM setup needed for this first run: `configs/yjmob-1k.yaml` ships with
`data/yjmob-1k/llm_diaries/validated_diaries_{weekday,weekend}.json` already
bundled in the project (`citybehavex init` writes them alongside this
README), and diary generation reuses a valid cache before ever calling an
LLM. These are real LLM-generated diaries adapted from a full-scale
CityBehavEx run, not placeholders.

To regenerate diaries instead of reusing the bundled ones -- e.g. after
changing `diaries.city_profile` or other diary-generation settings -- set
`CITYBEHAVEX_LLM_BASE_URL`, `CITYBEHAVEX_LLM_API_KEY`, and
`CITYBEHAVEX_LLM_MODEL` to a real OpenAI-compatible endpoint, then delete
the corresponding `validated_diaries_*.json` file (a changed config alone
won't invalidate the bundled cache, since it isn't fingerprinted against
`city_profile`). The CLI's `--start-aligners` starts the temporary local
aligner/embedding service for the simulation only; it's unrelated to the
diary LLM.

## About the downloaded comparison data

`citybehavex data download yjmob` fetches a small, entirely **synthetic**
baseline: the first 1,000 agents and first 7 days of an already-completed
CityBehavEx simulation run (see the bundled `PROVENANCE.md` after download).
It is not derived from the real YJMob100K challenge dataset, so it ships with
no data-license restrictions.

To instead validate against the real YJMob100K dataset, obtain it yourself
(it isn't redistributable) and run the preparation notebooks — see
"Setting up `data/` for the YJMOB scenario" in the main project README:
https://github.com/gefgu/citybehavex#setting-up-data-for-the-yjmob-scenario
