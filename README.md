# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mekii69/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mekii69/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Dilution-aware forecaster for GTA VI's PS5 trophy earn rates on Exophase at 2026-11-26T00:00:00Z. A trophy's rate is a ratio — earners over players tracked — and on launch night one refresh took the tracked count from 638 to 3,868 while almost no new trophies were earned, so every rate but the 100% one fell. This program models the two counts separately (power-law counts, momentum counts, logit trend, a cohort marginal-earn model, and a lag model where players earn a trophy some hours after being tracked) and ensembles them per trophy, with a spread from fit residuals, refresh-jump volatility, and model disagreement.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3 standard library only, no network, deterministic. Reads every `*.in` case in the input directory and writes one `*.out` forecast per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=9zhm5f8f&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
