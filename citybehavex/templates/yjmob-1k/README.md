# CityBehavEx YJMOB-1k example

1. Set `CITYBEHAVEX_LLM_BASE_URL`, `CITYBEHAVEX_LLM_API_KEY`, and `CITYBEHAVEX_LLM_MODEL`.
2. Run `citybehavex data download yjmob --project .`.
3. Run `citybehavex doctor --config configs/yjmob-1k.yaml`.
4. Run `citybehavex simulate --config configs/yjmob-1k.yaml --start-aligners`.

The LLM remains an externally managed OpenAI-compatible service.  The CLI starts
the temporary local aligner and embedding service for the simulation only.

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
