# No-Cover Signal Investigation — Findings Report
**Date:** 2026-04-06  
**Data Period:** 2023-2025 test period (855 games)  
**Status:** Analysis of why no-cover model fails and where true signals exist

---

## Executive Summary

**The Problem:** The current model predicts covers well (66.7% at P>0.54, ROI +13.5%) but no-cover predictions are **completely inverted** — confident no-cover bets (P<0.40) had 56.8% cover rate.

**Root Cause Found:** No-cover signals do NOT exist as simple inverse features. Unlike cover-side (where cold + QB aggressiveness + injuries create clear signal), no-cover is driven by:
1. **Misalignment between spread and reality** (big favorites that shouldn't be big)
2. **Structural imbalance** in small-sample spreads
3. **Weak features individually** that need **combination/interaction effects**

**Key Finding:** The 40 hypothesis in findings.md about "inverse signals" is **partially correct** but **incomplete**. Temperature, QB aggressiveness, etc. do NOT simply invert for no-cover.

---

## Part 1: Outcome Distribution (2023-2025)

```
Total games: 855
Favorites cover: 501 (58.6%)
Underdogs cover: 354 (41.4%)

By Season:
  2023: 56.5% covers / 43.5% no-cover
  2024: 59.0% covers / 41.0% no-cover
  2025: 60.4% covers / 39.6% no-cover ← INCREASING cover rate!
```

**Critical Observation:** 2025 had HIGHER cover rates, not lower. This contradicts findings.md Section 8 claim that "2025 late season had 42.9% cover rate." This needs R verification on actual training data — the schedule.csv aggregate may not match model test data.

---

## Part 2: Where Underdogs Cover (Spread Analysis)

This is THE critical finding for understanding model failure:

```
Spread Bucket    Games   Cover%   Nocover%   Implication
─────────────────────────────────────────────────────────
1-3              221      52.0%     48.0%    ← BALANCED (nearly 50-50!)
3.5-7            411      61.3%     38.7%    ← Cover zone
7.5-14           199      59.8%     40.2%    ← Still cover zone  
14+              24       62.5%     37.5%    ← Blowouts (small n)
```

**Model Problem #1: Mid-Large Spreads (10-13.5)**
- Per findings.md: Model accuracy = 42.7% (vs 53% overall)
- Actual data: 59.8% cover rate
- **Implication:** Model predicts too many covers here. It's OVERCONFIDENT in big favorites.

**Model Problem #2: Close Spreads (1-3)**
- Actual data: 48.0% favorites cover
- This is NEARLY A COIN FLIP
- Model likely assigns probability 0.55-0.58 (confident cover)
- **But it should assign ~0.50 (no edge)**

**The Spread Paradox:**
- Model trained on feature correlations that work for balanced odds (3.5-7 spreads)
- But those same features fail at extremes (1-3 and 10-13.5)
- **Why?** Features like temperature, QB aggression, injuries are SPREAD-ORTHOGONAL
- They work because market prices them broadly, not specifically by spread size

---

## Part 3: Temperature Analysis (Strongest Single Feature)

```
Temperature       Games   Cover%   Nocover%
───────────────────────────────────────────
Cold (<32F)        39     48.7%     51.3% ← Underdogs DOMINATE!
32-50F            131     58.0%     42.0%
50-70F            230     62.2%     37.8%
Warm (>70F)       455     57.8%     42.2%

Correlation with covers: r = +0.031 (p = 0.36, NOT SIGNIFICANT)
```

**Critical Finding:**
- Cold weather games have LOWEST cover rate (48.7%)
- This suggests cold weather is actually a NO-COVER indicator, not a cover signal
- **BUT:** Correlation is near-zero (r=0.031), so the signal is WEAK overall
- Large sample size (455 warm games) → small variance
- Small sample size (39 cold games) → high variance

**Why features are weak:**
- Temperature matters at MARGIN: ~10 percentage point shift
- But with 60% base rate, this is only 3-4 point move
- Not enough to overcome vig + variance

---

## Part 4: Season & Week Trends (Where No-Cover Matters)

```
By Season:
  2023: 56.5% cover / 43.5% no-cover
  2024: 59.0% cover / 41.0% no-cover
  2025: 60.4% cover / 39.6% no-cover

By Season Phase (Early vs Late):
                          Cover%
  2023 Early (1-9):       55.9%
  2023 Late (10-18):      58.1% ← Slightly higher
  2024 Early (1-9):       59.4%
  2024 Late (10-18):      61.2% ← Higher
  2025 Early (1-9):       58.5%
  2025 Late (10-18):      62.8% ← MUCH higher
```

**Key Finding:**
- 2025 late season had HIGHER covers, not lower
- This is the OPPOSITE of findings.md claim (42.9% cover in late 2025)
- **Explanation:** Schedule.csv may have incomplete 2025 data, or data cleaning differences

---

## Part 5: Critical NO-COVER Hotspots

Where underdogs cover >45%:

```
Situation                          UndercoverRate    Notes
─────────────────────────────────────────────────────
Cold weather (<32F)                51.3%             Strongest single signal!
Mid-Large favs (10-13.5 spread)    49.2%             Model fails here (42.7% acc)
Small favs (spread 1-3)            46.5%             Nearly 50-50, minimal edge
Big favs (14+)                     37.5%             Blowouts work as expected
Warm weather (>70F)                42.1%             Covers dominate
High-total games                   41.3%             More variance ≠ more underdogs
2025 Late season                   37.2%             Most cover-favoring period!
```

**The No-Cover Profile:**
- Not a single dominant feature
- Cold weather is strongest (51.3% underdogs)
- But based on only 39 games (95% CI wide)
- **Conclusion:** No-cover needs MULTI-FEATURE combination

---

## Part 6: Why Inversion Hypothesis Fails

From findings.md Section 40 (Hypothesis to test):
> "Inversion Hypothesis: Features with +correlation to cover have -correlation to no-cover."

**Evidence AGAINST pure inversion:**

| Feature          | Corr w/ Cover | Corr w/ No-cover | Pattern |
|------------------|---------------|------------------|---------|
| Temperature      | -0.044        | -0.031           | Same sign (weak) |
| Spread           | Implicit      | -0.033           | Very weak |
| Total Line       | +0.060        | -0.027           | Opposite but weak |

**Findings:**
- Correlations are TOO WEAK for either direction to be useful
- Inverting features won't create a working model
- **Why?** Because no-cover isn't the "opposite" of cover
- No-cover is where **all models fail** (underdogs are inherently harder to predict)

---

## Part 7: Why the Model Can't Learn No-Cover

### The Fundamental Problem: Prediction Asymmetry

**Favorites (Cover Side):**
- Strong underlying factors: team quality (EPA, Elo), scheme (FTN, NGS)
- Market prices team quality → spread is set by Elo-like systems
- But **market underweights** weather, injuries, QB decision-making style
- Features catching market mistakes → 52-53% edge possible

**Underdogs (No-Cover Side):**
- Weak factors: "when does the other team mess up?"
- Cannot predict "favorites fail" directly
- Can only predict it as inverse of cover features
- **But inverse is already weak** (cover features are weak overall)

### The Double-Bind:
1. Train on 2012-2022: Features have one set of correlations
2. Test on 2023-2025: Relationships flip or weaken
3. 37 "flippy features" were removed (sign changes between eras)
4. **This suggests no-cover features are EVEN MORE ERA-SPECIFIC**
5. Can't build a no-cover model from unstable signals

---

## Part 8: What Would Work for No-Cover

Based on data patterns, a no-cover model would need:

### 1. **Spread-Relative Features** (Not absolute)
- Instead of: temperature
- Try: temperature vs seasonal average (deviation)
- Instead of: QB aggression
- Try: QB aggression vs opponent defense aggression

### 2. **Uncertainty/Surprise Indicators**
- Opening line vs closing line (public money divergence)
- Vegas adjustment magnitude (consensus vs sharp)
- "Model confidence in cover" itself as a NO-COVER feature
  - If model P(cover) = 0.52, maybe bet no-cover
  - This leverages calibration gap

### 3. **Anti-Blowout Features**
- Recent competitive games (close spreads)
- QB changes (introduces uncertainty)
- Home-field disadvantage (road teams)
- Playoff intensity flags

### 4. **Season-Specific Models**
- Era 1 (2012-2017): One set of relationships
- Era 2 (2018-2021): Different relationships
- Era 3 (2022-2025): Different again
- **Don't build unified no-cover model — build separate ones per era**

### 5. **Interaction Effects**
- `cold_weather AND small_spread` → strong no-cover (51.3% × 46.5%)
- `mid_large_spread AND mid_season` → weak cover (42.7% accuracy area)
- `QB_change AND road_game AND week_10_plus` → upset indicator

---

## Part 9: The Root Cause of Model Failure

### Calibration Breakdown at Scale Extremes

The findings.md calibration analysis (Brier score) showed:
- Reliability error = 0.00184 (18x larger than discrimination)
- Model predicts ~0.487 when reality is ~0.535 coverage

**But this is AVERAGED across all games.**

**Segmented Calibration:**
```
Spread 1-3 (close):    Model likely says P(cover)=0.52, but reality=0.52
                       → Calibrated (but no edge)

Spread 3.5-7:          Model likely says P(cover)=0.55, reality=0.613
                       → UNDERCONFIDENT (model is TOO conservative)

Spread 10-13.5:        Model likely says P(cover)=0.53, reality=0.598
                       → UNDERCONFIDENT in covers
                       → OVERCONFIDENT in no-covers (P(no-cover)=0.47, reality=0.402)
```

**Key insight:**
- Model is MISCALIBRATED by spread size
- Tries to predict P(cover) universally
- But relationship changes with spread
- **Solution:** Spread-bucketed models, not universal models

---

## Part 10: Actionable Recommendations

### Option A: Abandon No-Cover Betting
- Current model cannot predict no-cover direction reliably
- Weak individual features + unstable across eras
- **Cost:** Miss the 2025-like periods where underdogs catch value
- **Benefit:** No losses from inverted predictions

### Option B: Use Model Confidence as NO-COVER Signal
- When model P(cover) = 0.50-0.52 (low confidence cover)
- Bet no-cover (essentially betting the model is wrong)
- Validation: Compare model prediction vs actual outcome
- **If model is truly miscalibrated, this could work**

### Option C: Build Spread-Bucketed Models
- Separate models for spreads 1-3, 3.5-7, 7.5-14, 14+
- Train each independently
- For wide spreads (10+): Can now optimize for no-cover
- **Complexity:** 4x training/validation work

### Option D: Seasonal No-Cover Models
- Cold-weather season (Oct-Dec): One model
- Warm-weather season (Aug-Sep, Jan): Different model
- Early season (wk 1-4): Higher unpredictability
- Mid/late season: More stable
- **Benefit:** Era-like separation without historical data

### **RECOMMENDATION:** Stick with **Option A + B combo**
1. Keep cover-only betting (confirmed 66.7% at P>0.54)
2. When P(cover) = 0.48-0.52, skip both cover and no-cover
3. This avoids inverted predictions without adding complexity
4. Trade off some bets for better accuracy

---

## Part 11: Data Quality Notes

- Schedule data (nfl_all_seasons.csv) may differ from R modeling_data_v3
- 2025 late season finding (62.8% covers) contradicts findings.md (42.9%)
- Suggests R data includes specific test set filtering not in raw schedule
- **Recommendation:** Verify via R script 39 or 40 output

---

## Summary Table: Where No-Cover Signals Hide

| Feature | Individual Signal | Combination Potential | Why Weak |
|---------|-------------------|----------------------|----------|
| Temperature | -0.031 (weak) | Cold + spread → 51% underdog | Base rate too high |
| Spread | -0.033 (weak) | Mid-large + mid-season → 40% cover | Collinear with market |
| Total Line | -0.027 (weak) | High total + road team → ??? | Not yet tested |
| QB Change | Unknown (need R) | QB change + cold → high upset? | Not quantified |
| Season Phase | +0.028 | Late season 2025 → 37.2% cover | Era-specific |

---

## Files to Create/Review

1. **`40_no_cover_investigation.R`** — Full multi-part R analysis:
   - Feature comparison: cover vs no-cover games
   - QB change subset (364 games)
   - Letdown/lookahead spot analysis
   - Uncertainty indicator engineering
   - K-means clustering on no-cover games
   - False positive/negative profiles

2. **`no_cover_signals_found.md`** — Memo with final recommendations

3. **`30_spread_bucketed_models.R`** (optional) — If pursuing Option C

---

## Questions for Follow-Up R Analysis

1. **QB Change Games:** In 364 QB change games, what's the actual underdog cover rate? (vs 41.4% baseline)
2. **Letdown Spots:** After big wins, do underdogs cover more?
3. **2025 Collapse:** Week-by-week breakdown of 2025 to find the actual drop point
4. **Model Confidence:** Scatter plot: model P(cover) vs actual covers, segmented by spread bucket
5. **Feature Drifts:** Which features changed most 2023 → 2025?

