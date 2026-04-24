# NFL Betting Model — Feature Signal Summary

## Top Signals Ranked by Predictive Power

### Tier 1: Composite Features (engineered from independent signals)

| # | Feature | Correlation | What it actually means |
|---|---------|-------------|----------------------|
| 1 | **cover_conviction** | **+0.082** | Weighted combination of all our best independent signals. Higher = the model's best evidence that the favorite will cover. Built from: cold weather + QB aggressiveness + defensive quality + blitz handling + turnovers + injuries + Elo gap. This is the most predictive single feature in the entire dataset. |
| 2 | **market_blindspot_score** | **+0.078** | Average of 5 signals the market doesn't price (cor with spread < 0.05 each): cold weather, QB aggressiveness, blitz rate, opponent injuries, dog QB sack vulnerability. These are things oddsmakers underweight. |

### Tier 2: Premium Data (NGS + FTN — the market barely prices these)

| # | Feature | Correlation | Spread Cor | What it actually means |
|---|---------|-------------|-----------|----------------------|
| 3 | **fav_ftn_roll5_rpo** | **-0.067** | 0.048 | The favorite's 5-game rolling RPO (Run-Pass Option) rate from FTN film charting. Negative correlation means: teams that run MORE RPOs tend to NOT cover. RPO-heavy offenses may be more predictable / less explosive against good defenses. The market barely prices this (0.048 cor with spread). |
| 4 | **fav_ngs_aggressiveness_roll5** | **+0.053** | -0.001 | Favorite QB's aggressiveness from Next Gen Stats (how often they throw into tight windows). More aggressive QBs cover MORE. This is essentially invisible to the market (0.001 cor with spread!). Aggressive QBs take shots downfield, which creates variance that benefits favorites who need to win by a margin. |
| 5 | **fav_ftn_roll5_blitz** | **+0.052** | -0.009 | Average blitz rate the favorite faces, from FTN charting. Counter-intuitive: teams that face MORE blitzes tend to cover MORE. Interpretation: if you're being blitzed a lot and still winning, you're handling pressure well — a sign of a well-coached offense the market hasn't fully priced. |
| 6 | **ngs_aggressiveness_roll5_diff** | **-0.049** | — | Difference in aggressiveness between fav and dog. Negative = when the dog's QB is MORE aggressive than the fav's, favorites tend to cover less. Aggressive underdogs take risks that sometimes pay off. |

### Tier 3: Universal Features (available 2012+, stable across eras)

| # | Feature | Correlation | What it actually means |
|---|---------|-------------|----------------------|
| 7 | **weather_temperature** | **-0.044** | Cold game temperature. Negative = colder games → favorites cover more. Cold weather suppresses passing offense, which increases variance and tends to favor the team with better coaching/preparation (usually the favorite). Our top stable universal feature. |
| 8 | **fav_def_success_rate_roll5** | **-0.042** | Favorite's defensive success rate (% of plays with positive EPA). Negative = LOWER success rate (worse defense) → more covers? Counter-intuitive. May reflect that defensive success rate mean-reverts hard, and teams with "bad" recent defense are underpriced. |
| 9 | **fav_injury_impact_score** | **+0.037** | Positional-weighted injury score for the opponent. Higher = more injuries on the other team → favorite covers more. QB injuries weighted 10x, O-line 3-5x, secondary 3-5x. The market knows about injuries but may not weight positional value correctly. |
| 10 | **turnover_diff** | **+0.036** | Recent 5-game turnover differential (fav - dog). Positive = favorite has been winning the turnover battle. Turnovers are high-variance and partly luck-driven, so teams with good recent turnover luck may be overvalued — but the positive correlation suggests the market UNDER-prices recent turnover trends. |
| 11 | **elo_diff** | **+0.031** | Elo rating differential (fav - dog). Higher Elo gap = more covers. But this has 0.50 correlation with spread — the market already prices most of this. The residual signal (~0.03) captures cases where Elo disagrees with the spread. |
| 12 | **dog_qb_sack_rate_roll5** | **-0.031** | Dog QB's sack rate over last 5 games. Higher sack rate for the underdog = more covers for the favorite. An immobile or indecisive underdog QB is a liability. |

### Tier 4: Engineered Composites (combinations)

| # | Feature | Correlation | What it actually means |
|---|---------|-------------|----------------------|
| 13 | **pressure_matchup** | **+0.044** | Combination of: fav QB aggressiveness + dog QB sack vulnerability + fav blitz handling. Captures the specific matchup of "decisive favorite QB vs vulnerable underdog QB." |
| 14 | **fresh_momentum** | **+0.042** | 3-game rolling offensive EPA + defensive EPA + turnovers. Captures teams that are HOT RIGHT NOW (more recent than 5-game average). Short-term form that the market may lag in pricing. |

---

## Key Insight: Why These Features Work

The common thread: **these features capture information that is orthogonal to the point spread.** The spread already incorporates team quality (EPA, Elo, win-loss record). Our edge comes from:

1. **Scheme tendencies** (FTN RPO rate, blitz handling) — game film data that most bettors don't analyze quantitatively
2. **QB decision-making style** (NGS aggressiveness) — a behavioral metric, not an outcome metric. The market prices outcomes (yards, TDs) but not decision-making process.
3. **Weather effects** — the market knows the weather but may not properly adjust the spread for how cold affects scoring patterns
4. **Injury granularity** — the market knows "QB is out" but may not properly weight the positional cascade (backup LT out is different from backup CB out)
5. **Recent form momentum** — 3-game rolling captures form changes faster than the market adjusts lines

## What DOESN'T Work

- **Rolling EPA / team quality metrics** — the spread already prices these better than we can
- **PFR pressure stats** — zero signal after the spread. The market knows about O-line quality.
- **No-cover predictions** — the model can't predict when favorites fail. Only bet covers.
- **Features that flip sign between eras** — 37 features were removed because their train/test relationship inverted. Non-stationary features are poison.
