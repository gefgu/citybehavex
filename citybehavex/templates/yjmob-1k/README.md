# CityBehavEx YJMOB-1k example

1. Set `CITYBEHAVEX_LLM_BASE_URL`, `CITYBEHAVEX_LLM_API_KEY`, and `CITYBEHAVEX_LLM_MODEL`.
2. Run `citybehavex data download yjmob --project .`.
3. Run `citybehavex doctor --config configs/yjmob-1k.yaml`.
4. Run `citybehavex simulate --config configs/yjmob-1k.yaml --start-aligners`.

The LLM remains an externally managed OpenAI-compatible service.  The CLI starts
the temporary local aligner and embedding service for the simulation only.
