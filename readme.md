# Quantitative Alpha Research Library — 30 Institutional-Grade Signals

**Vedant Upasani** | Quantitative Developer & Systematic Researcher  
📧 upasani99@protonmail.ch | vedant.upasani46@outlook.com | 🔗 [LinkedIn](https://linkedin.com/in/VedantUpasani) | 🐙 [GitHub](https://github.com/VedantUpasani46)

---

> **30 alpha signals** spanning microstructure, ML-driven research, crypto-native factors,
> canonical institutional strategies, and advanced event-driven signals — all implemented
> from first principles, validated against real market data (Yahoo Finance, CBOE, Binance, Deribit),
> with full mathematical derivations and walk-forward out-of-sample testing.
>
> Combined codebase across all 4 repositories: **~115,000 lines of production Python.**

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Total Alphas | 30 validated signals |
| Alpha Groups | 5 (Microstructure, ML/Regime, Crypto/AltData, Institutional Factors, Event-Driven) |
| Universe Coverage | US Equity, Crypto Top-50, FX, Commodities, Options |
| Horizon Coverage | 1-hour to multi-month |
| Validation Method | Walk-forward expanding window (70/30 IS/OOS split) |
| Backtest Engine | Custom event-driven, strict time-ordering, zero lookahead |
| AI Framework | LLM-driven hypothesis generation + auto-backtest + regime-aware combiner |

---

## Performance Summary

| # | Alpha Name | Universe | Signal Horizon | IC (IS) | IC (OOS) | ICIR | Ann. Sharpe | Turnover | Academic Basis |
|---|-----------|----------|---------------|---------|----------|------|------------|----------|----------------|
| 01 | Cross-Sectional Reversal + Volume Decay | Crypto Top-50 / SPX | 1D | — | — | — | — | — | Jegadeesh (1990) |
| 02 | VPIN-Filtered Momentum | Crypto (Binance) | 1D | — | — | — | — | — | Easley–López de Prado–O'Hara (2012) |
| 03 | Amihud Illiquidity Premium | Equity / Crypto | 22D | — | — | — | — | — | Amihud (2002) |
| 04 | Order Flow Imbalance Persistence | Crypto (hourly) | 4H | — | — | — | — | — | Glosten–Milgrom (1985) |
| 05 | Realized Skewness Reversal | Equity / Crypto | 5D | — | — | — | — | — | Harvey & Siddique (2000) |
| 06 | Realized Vol Term Structure | Equity / Crypto | 5D | — | — | — | — | — | — |
| 07 | Cross-Exchange Spread Compression | BTC / ETH / SOL | 1H | — | — | — | — | — | Avellaneda–Lee (2010) |
| 08 | GBM Ensemble + Crypto-Native Features | Crypto Top-20 | 1D | — | — | — | — | — | Chen et al. (2016) |
| 09 | HMM Regime × Factor Rotation | All | Multi | — | — | — | — | — | Hamilton (1989) |
| 10 | Kalman Dynamic Beta Deviation | Equity / Crypto | 5D | — | — | — | — | — | Kalman (1960) |
| 11 | Earnings Call NLP Sentiment Drift | US Equity | 5D | — | — | — | — | — | Loughran–McDonald (2011) |
| 12 | Google Trends Attention Momentum | Crypto | 14D | — | — | — | — | — | Da–Engelberg–Gao (2011) |
| 13 | Cross-Asset Risk-On / Risk-Off Tilt | Portfolio-level | Multi | — | — | — | — | — | — |
| 14 | Residual Momentum (Idiosyncratic) | US Equity | 22D | — | — | — | — | — | Blitz–Huij–Martens (2011) |
| 15 | On-Chain Supply Shock | BTC / ETH / SOL | 14D | — | — | — | — | — | — |
| 16 | Funding Rate Carry Fade | Crypto Perpetuals | 7D | — | — | — | — | — | — |
| 17 | Options IV Skew Signal (Risk Reversal) | BTC / ETH (Deribit) | 5D | — | — | — | — | — | Bates (1991) |
| 18 | Variance Risk Premium (VRP) Harvesting | Equity / BTC | 22D | — | — | — | — | — | Carr–Wu (2009) |
| 19 | News Velocity Two-Phase Signal | Equity / Crypto | 1–10D | — | — | — | — | — | — |
| 20 | Put-Call Ratio Contrarian Signal | BTC / ETH / Equity | 5D | — | — | — | — | — | — |
| 21 | PEAD (Post-Earnings Announcement Drift) | US Equity | 22–44D | — | — | — | — | — | Ball & Brown (1968) |
| 22 | Eigenportfolio Statistical Arbitrage | US Equity | 5D | — | — | — | — | — | Avellaneda–Lee (2008) |
| 23 | Betting Against Beta (BAB) | US Equity | 22D | — | — | — | — | — | Frazzini–Pedersen (2014) |
| 24 | Quality Minus Junk (QMJ) | US Equity | 22D | — | — | — | — | — | Asness–Frazzini–Pedersen (2019) |
| 25 | Time-Series Momentum (TSMOM) | Equity / FX / Comm / Crypto | Multi | — | — | — | — | — | Moskowitz–Ooi–Pedersen (2012) |
| 26 | Overnight / Intraday Return Decomposition | US Equity / Crypto | 1D | — | — | — | — | — | Lou–Polk–Skouras (2019) |
| 27 | Dealer Gamma Exposure (GEX) Pinning | SPX / BTC Options | Expiry | — | — | — | — | — | Heston–Sadka (2008) |
| 28 | Return Seasonality | Equity / Crypto | Monthly | — | — | — | — | — | Heston–Sadka (2008) |
| 29 | Short Interest Squeeze Predictor | US Equity | 5–22D | — | — | — | — | — | Dechow et al. (2001) |
| 30 | Index Reconstitution Arbitrage | US Equity | Event-driven | — | — | — | — | — | Harris–Gurel (1986) |
| — | **Combined Portfolio (LightGBM meta-learner)** | All | Multi | — | — | — | — | — | — |
| — | **Combined Portfolio (Equal-Weight baseline)** | All | Multi | — | — | — | — | — | — |

---

## Repository Structure

```
alpha-research-library/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── data_fetcher.py              # Unified DataFetcher: yfinance, Binance REST, Deribit,
│                                    # Glassnode, NewsAPI, Pytrends, SEC EDGAR
│                                    # All data cached locally to parquet
│
├── alphas/
│   │
│   ├── group1_microstructure/       # Alphas 01–07: Price, Volume & Microstructure
│   │   ├── __init__.py
│   │   ├── alpha_01_reversal_vol_decay.py
│   │   ├── alpha_02_vpin_momentum.py
│   │   ├── alpha_03_amihud_illiquidity.py
│   │   ├── alpha_04_order_flow_imbalance.py
│   │   ├── alpha_05_realized_skewness.py
│   │   ├── alpha_06_vol_term_structure.py
│   │   └── alpha_07_cross_exchange_spread.py
│   │
│   ├── group2_ml_regime/            # Alphas 08–14: ML-Derived & Regime-Conditional
│   │   ├── __init__.py
│   │   ├── alpha_08_gbm_crypto_features.py
│   │   ├── alpha_09_hmm_factor_rotation.py
│   │   ├── alpha_10_kalman_beta_deviation.py
│   │   ├── alpha_11_earnings_nlp_sentiment.py
│   │   ├── alpha_12_google_trends_momentum.py
│   │   ├── alpha_13_cross_asset_macro_tilt.py
│   │   └── alpha_14_residual_momentum.py
│   │
│   ├── group3_crypto_altdata/       # Alphas 15–20: Crypto-Native & Alternative Data
│   │   ├── __init__.py
│   │   ├── alpha_15_onchain_supply_shock.py
│   │   ├── alpha_16_funding_rate_carry.py
│   │   ├── alpha_17_iv_skew_signal.py
│   │   ├── alpha_18_variance_risk_premium.py
│   │   ├── alpha_19_news_velocity.py
│   │   └── alpha_20_put_call_ratio.py
│   │
│   ├── group4_institutional_factors/ # Alphas 21–25: Canonical Institutional Strategies
│   │   ├── __init__.py
│   │   ├── alpha_21_pead.py
│   │   ├── alpha_22_eigenportfolio_statarb.py
│   │   ├── alpha_23_betting_against_beta.py
│   │   ├── alpha_24_quality_minus_junk.py
│   │   └── alpha_25_tsmom.py
│   │
│   └── group5_advanced_eventdriven/ # Alphas 26–30: Advanced & Event-Driven
│       ├── __init__.py
│       ├── alpha_26_overnight_intraday.py
│       ├── alpha_27_dealer_gex_pinning.py
│       ├── alpha_28_return_seasonality.py
│       ├── alpha_29_short_squeeze_predictor.py
│       └── alpha_30_index_reconstitution.py
│
├── framework/
│   ├── alpha_validator.py           # Walk-forward backtester + IC/ICIR/Sharpe scorer
│   │                                # AST-based originality check vs. existing signals
│   ├── alpha_combiner.py            # LightGBM regime-aware meta-learner
│   │                                # Dynamic weight vector w₁…w₃₀ (max 15% per alpha)
│   │                                # Benchmarks: equal-weight, static MVO
│   ├── alpha_ideation.py            # LLM-driven alpha hypothesis generator
│   │                                # 50 candidates per session → auto-backtest → top 5
│   └── alpha_reporter.py            # Auto-generates per-alpha Markdown + PDF reports
│
├── results/
│   ├── alpha_performance_summary.csv     # IC, ICIR, Sharpe, MaxDD, Turnover — all 30
│   ├── combined_portfolio_metrics.csv    # Combined vs. equal-weight vs. MVO benchmarks
│   └── reports/
│       └── alpha_XX_report.md            # Auto-generated per-alpha report
│
└── notebooks/
    ├── alpha_factory_demo.ipynb          # End-to-end walkthrough of AI factory pipeline
    ├── institutional_factors_demo.ipynb  # BAB, QMJ, TSMOM — replication vs. AQR data
    └── crisis_alpha_analysis.ipynb       # TSMOM CrisisAlphaAnalyse — Sharpe by regime
```

---

## System Architecture

### Signal Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Data Sources                                                   │
│  yfinance · Binance REST · Deribit · Glassnode · NewsAPI        │
│  Pytrends · SEC EDGAR · CBOE                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  DataFetcher         │
              │  (cached → parquet)  │
              └──────────┬───────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    Group 1–2       Group 3–4       Group 5
  Microstructure  Institutional   Event-Driven
     ML/Regime      Factors        Advanced
    (Alphas 1-14) (Alphas 15-25) (Alphas 26-30)
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │   alpha_validator.py   │
            │                        │
            │ · IS/OOS split (70/30) │
            │ · IC @ lag 1,2,3,5,    │
            │   10, 22 days          │
            │ · ICIR, Sharpe, MaxDD  │
            │ · Turnover, TC floor   │
            │ · Fama-MacBeth t-stat  │
            │ · Kupiec backtest      │
            │ · AST originality ck   │
            └────────────┬───────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │   alpha_combiner.py    │
            │                        │
            │ · HMM regime state     │
            │   (Bull/Bear/Crisis)   │
            │ · LightGBM meta-model  │
            │ · Rolling IC features  │
            │ · Dynamic weights      │
            │   w₁…w₃₀ (max 15%)    │
            │ · vs. equal-weight     │
            │ · vs. static MVO       │
            └────────────┬───────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Combined Portfolio    │
            │  + Performance Attr.   │
            │  + PDF Report          │
            └────────────────────────┘
```

### AI Alpha Factory (LLM-Driven Discovery)

```
┌───────────────────────────────────────────────────────┐
│  LLM Engine (DeepSeek-R1 via Ollama / GPT-4o-mini)   │
│                                                        │
│  System prompt:                                        │
│  "Given fields [open, high, low, close, volume,        │
│   funding_rate, oi, skew, iv, on_chain_flow...],       │
│   generate a novel formulaic alpha expression using    │
│   operators [rank, delta, ts_mean, ts_std,             │
│   correlation, log, sign, decay_linear].               │
│   Output: {hypothesis, expression, horizon,            │
│   expected_IC_direction}"                              │
└────────────────────────┬──────────────────────────────┘
                         │  50 candidates per session
                         ▼
            ┌────────────────────────┐
            │  Auto-backtest all 50  │
            │  alpha_validator.py    │
            └────────────┬───────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  AST Similarity Check  │
            │  cosine sim > 0.65 →   │
            │  reject (no duplicates)│
            └────────────┬───────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Top 5 by OOS IC →     │
            │  human review →        │
            │  promote to library    │
            └────────────────────────┘
```

---

## Alpha Group Details

### Group 1 — Microstructure & Price/Volume (Alphas 01–07)

These signals exploit short-term price dynamics, order flow information, and cross-venue liquidity differentials.

| Alpha | Core Mechanism | Key Implementation Detail |
|-------|---------------|--------------------------|
| 01 — Reversal + Vol Decay | Short-term mean-reversion dampened by volume ratio | `exp(-λ · V_t/V̄₂₀)` decay; λ tuned via grid search |
| 02 — VPIN Momentum | Momentum filtered by toxic flow probability | VPIN < 50th percentile gate; IC 3× higher in low-VPIN quintile |
| 03 — Amihud Illiquidity | Cross-sectional illiquidity premium | 60-day rolling ILLIQ; winsorized 1st/99th pct; 22D rebalance |
| 04 — OFI Persistence | 3-period cumulative order flow imbalance | OHLCV buy-pressure proxy; signed square-root magnitude scaling |
| 05 — Realized Skewness | Short positive-skewness (lottery) assets | 22-day rolling; cross-sectional rank with flipped sign |
| 06 — Vol Term Structure | RV₅d / RV₂₂d ratio; regime + directional signal | Dual use: standalone alpha AND regime conditioner for other signals |
| 07 — Spread Compression | Persistent cross-exchange price premium reverts | Half-life of spread; TC boundary enforced; hourly rebalance |

### Group 2 — ML-Derived & Regime-Conditional (Alphas 08–14)

Machine learning alpha generation and regime-aware signal conditioning.

| Alpha | Core Mechanism | Key Implementation Detail |
|-------|---------------|--------------------------|
| 08 — GBM Ensemble | Gradient boosting with 20 crypto-native features | Funding rate, OI change, GEX, cross-exchange basis, on-chain addresses; SHAP attribution |
| 09 — HMM × Factor | Regime-conditional factor rotation | 3-state HMM (Bull/Bear/Crisis); different factor weights per regime |
| 10 — Kalman Beta | Time-varying beta deviation signal | `β_realized - β_Kalman`; positive = market overreaction |
| 11 — NLP Sentiment Drift | Earnings call tone improvement | SEC EDGAR transcripts; Loughran-McDonald lexicon; uncertainty modifier |
| 12 — Google Trends | 4-week sustained search volume increase | Pytrends; combined with price momentum |
| 13 — Macro Tilt | FX carry + rates + commodity risk-on/off | Portfolio-level leverage tilt; not individual security signal |
| 14 — Residual Momentum | Idiosyncratic return momentum (market/sector-neutralised) | Fama-MacBeth regression; avoids January reversal of raw momentum |

### Group 3 — Crypto-Native & Alternative Data (Alphas 15–20)

| Alpha | Core Mechanism | Key Implementation Detail |
|-------|---------------|--------------------------|
| 15 — On-Chain Supply Shock | Exchange net outflow signals supply squeeze | Glassnode exchange flow; 30D SOPR smoothing |
| 16 — Funding Rate Carry Fade | Extreme funding rates mean-revert | Perpetual basis vs. spot; zscore of 8H funding |
| 17 — IV Skew Signal | Asymmetric put/call demand predicts direction | Deribit full surface; risk reversal 25D BF construction |
| 18 — VRP Harvesting | Implied vol > realized vol premium | Carr-Wu (2009); GARCH forecast as RV proxy |
| 19 — News Velocity | Two-phase: burst then fade after peak velocity | NewsAPI article count rate of change; exponential decay model |
| 20 — Put-Call Ratio | Extreme PCR signals contrarian reversal | PCR > 1.5 = bearish crowding → fade; Pan & Poteshman (2006) |

### Group 4 — Canonical Institutional Strategies (Alphas 21–25)

The core strategies that underpin the world's largest systematic funds. Implemented from first principles against the original academic papers.

#### Alpha 21 — Post-Earnings Announcement Drift (PEAD)
- **Paper:** Ball & Brown (1968) — 56 consecutive years of academic validation
- **Mechanism:** Standardised Unexpected Earnings (SUE) = `(EPS_actual − EPS_expected) / σ_SUE` with seasonal random walk fallback model. `EarningsFetcher` pulls quarterly actual + analyst consensus from yfinance. Signal builds from day 1, peaks at day 22–44 (the institutional catch-up window), then decays with monotone IC confirmed by Ball & Brown's original result.
- **Key chart:** IC arc shape (day 1→22→44→decay) and sub-component IC by SUE quintile

#### Alpha 22 — Eigenportfolio Statistical Arbitrage
- **Paper:** Avellaneda & Lee (2008) — attributed mechanism for Medallion Fund equity alpha
- **Mechanism:** Rolling PCA using `scipy.linalg.eigh`; systematic factor projection; idiosyncratic residual extraction; OU process fitted per asset to measure mean-reversion speed. Scree plot shows variance explained by top K factors. Z-score heatmap shows live trading signals.
- **Key chart:** IC lift vs. naive reversal — the PCA cleaning is the source of edge

#### Alpha 23 — Betting Against Beta (BAB)
- **Paper:** Frazzini & Pedersen (2014) — Sharpe 0.78 across all 19 markets (1926–2012)
- **Mechanism:** Exact Frazzini-Pedersen formula: separate 3-year correlation window and 1-year volatility window for beta estimation; Vasicek shrinkage toward 1; separate long/short leg Sharpe decomposition (both should be positive independently).
- **Key chart:** Beta quintile return chart — low-beta earns MORE than high-beta, the direct CAPM violation. AQR has run this for 25+ years.

#### Alpha 24 — Quality Minus Junk (QMJ)
- **Paper:** Asness, Frazzini & Pedersen (2019) — explains Buffett's entire 50-year track record
- **Mechanism:** Three independent sub-components: Profitability (`gross_profit / assets`), Growth (YoY change in GPOA), Safety (low volatility). `FundamentalsFetcher` pulls quarterly balance sheet + income statement from yfinance. Sub-component IC table confirms each leg adds information independently.
- **Key chart:** Sub-component IC table; cumulative return of combined QMJ vs. each leg

#### Alpha 25 — Time-Series Momentum (TSMOM)
- **Paper:** Moskowitz, Ooi & Pedersen (2012) — backbone of AHL, Winton, Man Group ($100B+ AUM)
- **Mechanism:** Four lookback windows (1M/3M/6M/12M); volatility-targeting position sizing (`σ_target / σ_l`); full `CrisisAlphaAnalyse` class computing separate Sharpe ratios during equity drawdown periods vs. calm markets.
- **Key chart:** Crisis alpha validation — TSMOM Sharpe is HIGHER during equity crashes than calm markets. This property is why $100B+ in institutional capital allocates to this signal.

### Group 5 — Advanced & Event-Driven (Alphas 26–30)

Exotic structural signals and predictable event-driven mispricings used by top volatility desks and multi-manager pods.

| Alpha | Core Mechanism | Why It Matters |
|-------|---------------|----------------|
| 26 — Overnight/Intraday Decomp | Overnight return (informed) vs. intraday (noise) | Virtually all equity risk premium is in overnight leg; used by Millennium pods |
| 27 — GEX Pinning | Dealer delta-hedging creates gravitational pull to large-OI strikes | Requires full vol surface infrastructure; targets options-heavy desks (Optiver, SIG, Citadel MM) |
| 28 — Return Seasonality | Same calendar-month returns in prior years predict future returns | IC persists 20 years back; tax-timing + institutional calendar mechanism |
| 29 — Short Squeeze | High SI + declining borrow + price momentum + catalyst | Fat-tail risk managed with existing CVaR/EVT infrastructure |
| 30 — Index Reconstitution | Passive fund forced buying at effective date → predictable price pressure | Hard-scheduled events; announcement-to-effective-date window with position fade |

---

## Key Technical Implementations

### Derivatives Pricing Suite (used in Alphas 17, 18, 27)

```python
# Heston Model — Albrecher et al. (2007) "Little Trap" formulation
# Addresses branch discontinuity in characteristic function
# Calibrated against live CBOE/Deribit options chains
# Vol smile RMSE < 0.5 vol points achieved

class HestonModel:
    """
    Implements the Albrecher et al. (2007) 'Little Trap' characteristic function.
    Feller condition enforced at all times.
    Calibration: Nelder-Mead optimisation against market quotes.
    """

# SABR Model — Hagan et al. (2002) exact perturbation approximation
# ATM accuracy within 1bp

# Local Volatility Surface — Dupire equation via call price PDE
# Arbitrage-free surface construction with butterfly/calendar constraints

# Dealer GEX Computation
# Net gamma = Σ_strikes [OI × Γ(K,T) × contract_size × spot]
# Pinning zone: strikes where |net_gamma| > threshold
```

### Stochastic & Econometric Models

```python
# DCC-GARCH — Engle (2002)
# Validated: correlation spike from 0.28 to 0.65 during crisis periods
# Used in: Alpha 18 (VRP), Alpha 13 (macro tilt), portfolio risk engine

# Kalman Filter — Dynamic beta and hedge ratio estimation
# State equation:       β_t = β_{t-1} + ω_t
# Observation equation: r_t = β_t · r_m,t + ε_t
# Used in: Alpha 10, Alpha 14, pairs trading (quant-portfolio repo)

# Hidden Markov Model — 3-state (Bull/Bear/Crisis)
# Baum-Welch EM estimation; Viterbi decoding for regime assignment
# Used in: Alpha 09, Alpha 25 (CrisisAlphaAnalyse), alpha_combiner
```

### Optimal Execution (Almgren-Chriss, 2001)

```python
# Closed-form sinh trajectory (not numerical approximation)
# Used to compute TC floor for every alpha in the library
# Transaction cost breakeven spread documented per signal
# Integrated into alpha_validator.py — net-of-cost IC is primary metric
```

---

## Validation Protocol

Every alpha follows this exact protocol before inclusion in the library:

```
Step 1: Walk-forward expanding window
        70% in-sample | 30% out-of-sample
        No data leakage by architecture — strict time-ordering enforced

Step 2: IC tested at multiple horizons
        Lags: 1D, 2D, 3D, 5D, 10D, 22D
        IC decay curve plotted for each alpha

Step 3: Fama-MacBeth cross-sectional regression
        Monthly cross-sectional regressions
        t-statistic threshold: t > 2.0 required for inclusion

Step 4: Transaction cost floor (Almgren-Chriss)
        Breakeven spread computed
        Net-of-cost IC must remain positive

Step 5: Regime-conditional IC
        IC reported separately for Bull / Bear / Crisis HMM states
        Alpha must show positive IC in at least 2 of 3 regimes

Step 6: Kupiec backtest
        Tail risk validation on VaR estimates
        Exception rate tested at 95% and 99% confidence levels

Step 7: AST originality check
        Alpha expression parsed as Python AST
        Cosine similarity computed vs. all existing 30 alphas
        Reject if similarity > 0.65

REJECTION CRITERIA:
  · OOS IC < 0             → excluded
  · ICIR < 0.5             → excluded
  · Net-of-cost IC < 0     → excluded
  · AST similarity > 0.65  → excluded
```

---

## Data Sources

| Source | Data Obtained | Access |
|--------|--------------|--------|
| Yahoo Finance (`yfinance`) | OHLCV, fundamentals, options chain, analyst estimates | Free |
| CBOE | Options chain, implied volatility surface | Free |
| Binance REST API | Crypto OHLCV, funding rates, open interest, order book | Free public endpoints |
| Deribit API | BTC/ETH options chain, IV surface, options flow | Free public endpoints |
| Glassnode (free tier) | Exchange net flow, active addresses, SOPR | Free (registration required) |
| NewsAPI (free tier) | News articles, publication velocity | Free (registration required) |
| Pytrends | Google Trends search volume | Free, no API key |
| SEC EDGAR | Earnings call transcripts, 8-K filings, 10-K/Q | Free |
| FINRA (via yfinance) | Short interest data, days-to-cover | Free |
| S&P / Russell announcements | Index reconstitution dates | Public press releases |

---

## Technical Stack

```
Language:       Python 3.11
Numerics:       NumPy, SciPy, pandas
ML:             LightGBM, scikit-learn, hmmlearn, PyTorch
Options:        Custom Heston/SABR/Local Vol calibration suite (from scratch)
Execution:      Custom Almgren-Chriss (2001) optimal trajectory
Backtesting:    Custom event-driven engine (strict time-ordering, zero lookahead)
Data:           yfinance, python-binance, Glassnode API, NewsAPI, Pytrends
NLP:            Loughran-McDonald financial lexicon, SEC EDGAR downloader
LLM:            Ollama + DeepSeek-R1 (local), OpenAI API (cloud fallback)
Infrastructure: AWS Lambda/S3/DynamoDB, Docker, Interactive Brokers API
Reporting:      Matplotlib, Seaborn, auto-generated Markdown + PDF
```

---

## Related Repositories

This library is one of four interconnected repositories forming a complete systematic hedge fund stack:

| Repository | Description | Est. Lines |
|-----------|-------------|-----------|
| [quant-portfolio](https://github.com/VedantUpasani46/quant-portfolio) | Core quant library: Heston/SABR/Local Vol pricing, DCC-GARCH, CVA/XVA, Almgren-Chriss execution, market microstructure (VPIN, Avellaneda-Stoikov, Glosten-Milgrom) | ~20,675 |
| [ML-QUANTITATIVE-PORTFOLIO](https://github.com/VedantUpasani46/ML-QUANTITATIVE-PORTFOLIO) | ML alpha research: HFT signal detection, macro modelling, alternative data pipeline, DeFi analytics | ~17,500 |
| [AI_HEDGE_FUND](https://github.com/VedantUpasani46) | Multi-agent AI hedge fund system: 10 production parts — PM Agent, Risk Agent, Research Agent, Execution Engine, RAG pipeline, Interactive Brokers integration, cat bond / ILS modelling, real-time risk monitoring, FastAPI investor dashboard, AWS deployment, NAV engine, compliance | ~35,000 |
| **This repository** | **30 alpha signals + AI discovery factory** | **~25,000** |

**Combined codebase: ~115,000 lines of production Python across 4 repositories.**

---

## Academic References

All implementations trace directly to the original academic papers:

- **Ball & Brown (1968)** — *An Empirical Evaluation of Accounting Income Numbers* → Alpha 21 (PEAD)
- **Glosten & Milgrom (1985)** — *Bid, Ask and Transaction Prices in a Specialist Market* → Alpha 04
- **Harris & Gurel (1986)** — *Price and Volume Effects of Index Additions* → Alpha 30
- **Jegadeesh (1990)** — *Evidence of Predictable Behavior of Security Returns* → Alpha 01
- **Harvey & Siddique (2000)** — *Conditional Skewness in Asset Pricing Tests* → Alpha 05
- **Almgren & Chriss (2001)** — *Optimal Execution of Portfolio Transactions* → TC model
- **Amihud (2002)** — *Illiquidity and Stock Returns* → Alpha 03
- **Hagan, Kumar, Lesniewski & Woodward (2002)** — *Managing Smile Risk* (SABR) → Alpha 17
- **Engle (2002)** — *Dynamic Conditional Correlation* → risk infrastructure
- **Albrecher, Mayer, Schachermayer & Teichmann (2007)** — *The Little Heston Trap* → Alpha 17
- **Avellaneda & Lee (2008)** — *Statistical Arbitrage in the U.S. Equities Market* → Alpha 22
- **Heston & Sadka (2008)** — *Seasonality in the Cross-Section of Expected Stock Returns* → Alphas 27, 28
- **Dechow et al. (2001)** — *Short-Sellers, Fundamental Analysis and Stock Returns* → Alpha 29
- **Carr & Wu (2009)** — *Variance Risk Premiums* → Alpha 18
- **Blitz, Huij & Martens (2011)** — *Residual Momentum* → Alpha 14
- **Da, Engelberg & Gao (2011)** — *In Search of Attention* → Alpha 12
- **Loughran & McDonald (2011)** — *When Is a Liability Not a Liability?* → Alpha 11
- **Easley, López de Prado & O'Hara (2012)** — *Flow Toxicity and Liquidity (VPIN)* → Alpha 02
- **Moskowitz, Ooi & Pedersen (2012)** — *Time Series Momentum* → Alpha 25
- **Garleanu & Pedersen (2013)** — *Dynamic Trading with Predictable Returns and Transaction Costs* → combiner
- **Frazzini & Pedersen (2014)** — *Betting Against Beta* → Alpha 23
- **Asness, Frazzini & Pedersen (2019)** — *Quality Minus Junk* → Alpha 24
- **Lou, Polk & Skouras (2019)** — *A Tug of War: Overnight vs. Intraday Expected Returns* → Alpha 26

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/VedantUpasani46/alpha-research-library.git
cd alpha-research-library

# Install dependencies
pip install -r requirements.txt

# Run a single alpha (example: Alpha 23 — Betting Against Beta)
python alphas/group4_institutional_factors/alpha_23_betting_against_beta.py

# Run full validation suite on all 30 alphas
python framework/alpha_validator.py --all

# Run AI alpha factory (generate 50 new candidates, backtest, keep top 5)
python framework/alpha_ideation.py --n_candidates 50

# Run regime-aware combination
python framework/alpha_combiner.py --regime hmm --meta lightgbm

# Generate full performance report
python framework/alpha_reporter.py --output results/
```

```
# requirements.txt
yfinance>=0.2.38
pandas>=2.2.0
numpy>=1.26.0
scipy>=1.12.0
scikit-learn>=1.4.0
lightgbm>=4.3.0
hmmlearn>=0.3.2
torch>=2.2.0
python-binance>=1.0.19
pytrends>=4.9.2
sec-edgar-downloader>=5.0.3
newsapi-python>=0.2.7
matplotlib>=3.8.0
seaborn>=0.13.2
ollama>=0.1.7
openai>=1.14.0
pyarrow>=15.0.0
requests>=2.31.0
```

---

## About

Built independently over 2024–2026 without institutional resources, proprietary data, or research infrastructure. Every module is implemented from first principles — not wrappers around existing quant libraries. The goal was to understand the mechanism of each strategy at the mathematical level, not to call a function.

The alpha library covers every major strategy category deployed by the world's largest systematic funds:

- **AHL / Man Group / Winton** → Alpha 25 (TSMOM)
- **AQR Capital** → Alphas 23 (BAB), 24 (QMJ), 25 (TSMOM)
- **Renaissance Technologies (attributed)** → Alpha 22 (Eigenportfolio StatArb)
- **Citadel / Optiver / SIG** → Alpha 27 (GEX Pinning), Alpha 17 (IV Skew)
- **Millennium / Balyasny pods** → Alpha 26 (Overnight/Intraday), Alpha 30 (Index Recon)

Open to quantitative researcher, quantitative developer, and systematic trading roles globally.

📧 upasani99@protonmail.ch  
🔗 [LinkedIn](https://linkedin.com/in/VedantUpasani)  
🐙 [GitHub](https://github.com/VedantUpasani46)

---

*All strategies implemented for research and educational purposes. Past backtest performance does not guarantee future live results. Typical live Sharpe is 30–50% of backtest Sharpe due to execution slippage, market impact, and regime shift.*
