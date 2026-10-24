import type { IndicatorSettings, EngineSettings, Decision, LayerResult } from '@/types';

// ---------------------------------------------------------------------------
// 4-layer decision pipeline
//
//   Layer 1  EP-EL value filter     is the (estimated probability x payout) worth the bet?
//   Layer 2  False-signal filter    is this particular signal likely to be a whipsaw?
//   Layer 3  Fractional Kelly       how much to stake - only for an edge already demonstrated
//   Layer 4  Volatility/regime gate is the market in a state where the signal is reliable?
//
// Evaluation order: raw signal -> L4 -> L2 -> L1 -> L3. Value and signal checks
// run BEFORE the staking calculation, so Kelly only ever sizes a trade that has
// already been accepted; it never turns a rejected bet into an acceptable one.
//
// The probability is NOT assumed. It is measured: the same signal rule is replayed
// over the tick history already in memory and each past signal is scored against
// what price actually did `horizonTicks` later (a tie is a loss, like Rise/Fall).
// ---------------------------------------------------------------------------

export const DEFAULT_ENGINE: EngineSettings = {
  enabled: true,
  payoutRatio: 0.95,
  minEv: 0.01,
  horizonTicks: 5,
  minSamples: 40,
  maxFalseRate: 0.5,
  persistTicks: 3,
  minStrength: 0.3,
  stakingMode: 'kelly',
  kellyFraction: 0.25,
  kellyZ: 1.64,
  maxStakePct: 2,
  minStake: 0.35,
  regimeEnabled: true,
  volRatioMin: 0.5,
  volRatioMax: 2.0,
  minEfficiency: 0.1,
};

// ----- indicator helpers (all causal: value at i only uses prices[0..i]) -----

function ema(prices: number[], period: number): number[] {
  const k = 2 / (period + 1);
  const out: number[] = [];
  let v = prices[0];
  for (let i = 0; i < prices.length; i++) {
    v = i === 0 ? prices[0] : prices[i] * k + v * (1 - k);
    out.push(v);
  }
  return out;
}

function rsi(prices: number[], period: number): number[] {
  const out: number[] = new Array(prices.length).fill(50);
  if (prices.length < period + 1) return out;
  let g = 0, l = 0;
  for (let i = 1; i <= period; i++) {
    const c = prices[i] - prices[i - 1];
    if (c > 0) g += c; else l -= c;
  }
  g /= period; l /= period;
  out[period] = l === 0 ? 100 : 100 - 100 / (1 + g / l);
  for (let i = period + 1; i < prices.length; i++) {
    const c = prices[i] - prices[i - 1];
    g = (g * (period - 1) + (c > 0 ? c : 0)) / period;
    l = (l * (period - 1) + (c < 0 ? -c : 0)) / period;
    out[i] = l === 0 ? 100 : 100 - 100 / (1 + g / l);
  }
  return out;
}

/** Rolling mean and population std-dev over `period`, causal. */
function rolling(prices: number[], period: number) {
  const mean: number[] = [];
  const std: number[] = [];
  let sum = 0, sumSq = 0;
  for (let i = 0; i < prices.length; i++) {
    sum += prices[i]; sumSq += prices[i] * prices[i];
    if (i >= period) { sum -= prices[i - period]; sumSq -= prices[i - period] ** 2; }
    const n = Math.min(i + 1, period);
    const m = sum / n;
    mean.push(m);
    std.push(Math.sqrt(Math.max(0, sumSq / n - m * m)));
  }
  return { mean, std };
}

/** Equal-weight running mean of the buffer. Ticks carry no volume, so this is the
 *  honest tick-data stand-in for VWAP (the old code used random volumes). */
function runningMean(prices: number[]): number[] {
  const out: number[] = [];
  let s = 0;
  for (let i = 0; i < prices.length; i++) { s += prices[i]; out.push(s / (i + 1)); }
  return out;
}

/** Lower bound of the Wilson score interval for a proportion p observed over n trials. */
function wilsonLower(p: number, n: number, z: number): number {
  if (n <= 0) return 0;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const centre = p + z2 / (2 * n);
  const margin = z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n));
  return Math.max(0, (centre - margin) / denom);
}

interface Series {
  prices: number[];
  emaS: number[];
  emaL: number[];
  rsiV: number[];
  bbMean: number[];
  bbStd: number[];
  vwap: number[];
}

function buildSeries(prices: number[], ind: IndicatorSettings): Series {
  const bb = rolling(prices, ind.bbPeriod);
  return {
    prices,
    emaS: ema(prices, ind.emaShort),
    emaL: ema(prices, ind.emaLong),
    rsiV: rsi(prices, ind.rsiPeriod),
    bbMean: bb.mean,
    bbStd: bb.std,
    vwap: runningMean(prices),
  };
}

/** Base signal at index i: +1 CALL, -1 PUT, 0 none. Same rule used live and in the replay. */
function rawSignal(s: Series, i: number, ind: IndicatorSettings): 1 | -1 | 0 {
  const gap = s.emaS[i] - s.emaL[i];
  if (gap === 0) return 0;
  const dir = gap > 0 ? 1 : -1;
  const r = s.rsiV[i];
  if (!(r > 30 && r < 70)) return 0;
  if (ind.bbEnabled && i >= ind.bbPeriod - 1) {
    const width = (2 * ind.bbDeviation * s.bbStd[i]) / s.bbMean[i];
    if (!(width > 0.001)) return 0;
  }
  if (ind.vwapEnabled) {
    // direction must agree with which side of the running mean price is on
    if (dir === 1 && !(s.prices[i] > s.vwap[i])) return 0;
    if (dir === -1 && !(s.prices[i] < s.vwap[i])) return 0;
  }
  return dir;
}

function avgAbsMove(prices: number[], end: number, window: number): number {
  const start = Math.max(1, end - window + 1);
  let s = 0, n = 0;
  for (let i = start; i <= end; i++) { s += Math.abs(prices[i] - prices[i - 1]); n++; }
  return n ? s / n : 0;
}

function stdOfMoves(prices: number[], end: number, window: number): number {
  const start = Math.max(1, end - window + 1);
  const moves: number[] = [];
  for (let i = start; i <= end; i++) moves.push(prices[i] - prices[i - 1]);
  if (moves.length < 2) return 0;
  const m = moves.reduce((a, b) => a + b, 0) / moves.length;
  return Math.sqrt(moves.reduce((a, b) => a + (b - m) ** 2, 0) / moves.length);
}

function efficiencyRatio(prices: number[], end: number, window: number): number {
  const start = end - window;
  if (start < 0) return 0;
  let path = 0;
  for (let i = start + 1; i <= end; i++) path += Math.abs(prices[i] - prices[i - 1]);
  return path === 0 ? 0 : Math.abs(prices[end] - prices[start]) / path;
}

export interface EngineContext {
  bankroll: number;       // current account balance, used by Kelly
  fallbackStake: number;  // stake to use when staking mode is fixed / martingale
}

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

export function evaluate(
  prices: number[],
  ind: IndicatorSettings,
  cfg: EngineSettings,
  ctx: EngineContext
): Decision {
  const H = Math.max(1, Math.round(cfg.horizonTicks));
  const b = cfg.payoutRatio;
  const breakeven = 1 / (1 + b);
  const layers: LayerResult[] = [];
  const metrics: Decision['metrics'] = {
    pHat: null, pLower: null, breakeven, ev: null, kelly: null,
    samples: 0, volRatio: null, efficiency: null,
  };
  const skip = (reason: string): Decision => ({
    action: 'skip', direction: null, stake: 0, reason, layers, metrics,
  });

  if (!ind.enabled) {
    return skip('Indicators are disabled - enable them to generate signals');
  }

  const minData = Math.max(ind.emaLong, ind.bbPeriod, ind.rsiPeriod, 100) + 5;
  if (prices.length < minData) {
    return skip(`Warming up: ${prices.length}/${minData} ticks`);
  }

  const last = prices.length - 1;
  const s = buildSeries(prices, ind);

  // ---- raw signal -------------------------------------------------------
  const sig = rawSignal(s, last, ind);
  layers.push({
    id: 'signal', name: 'Base signal', pass: sig !== 0,
    detail: sig === 0 ? 'No EMA/RSI/BB/VWAP agreement' : `${sig === 1 ? 'CALL' : 'PUT'} (EMA ${ind.emaShort}/${ind.emaLong})`,
  });
  if (sig === 0) return skip('No base signal');
  const direction = sig === 1 ? 'CALL' : 'PUT';

  // Pipeline switched off: trade every base signal with the fixed / martingale stake.
  if (!cfg.enabled) {
    layers.push({ id: 'regime', name: 'Pipeline', pass: true, skipped: true, detail: 'Disabled - trading every base signal' });
    return { action: 'trade', direction, stake: ctx.fallbackStake, reason: 'Pipeline disabled', layers, metrics };
  }

  // ---- Layer 4: volatility / regime gate (cheap, so it runs first) --------
  const shortSd = stdOfMoves(prices, last, 20);
  const longSd = stdOfMoves(prices, last, 100);
  const volRatio = longSd > 0 ? shortSd / longSd : 0;
  const eff = efficiencyRatio(prices, last, 20);
  metrics.volRatio = volRatio;
  metrics.efficiency = eff;
  if (cfg.regimeEnabled) {
    const volOk = volRatio >= cfg.volRatioMin && volRatio <= cfg.volRatioMax;
    const effOk = eff >= cfg.minEfficiency;
    layers.push({
      id: 'regime', name: 'L4 Regime gate', pass: volOk && effOk,
      detail: `vol ratio ${volRatio.toFixed(2)} (${cfg.volRatioMin}-${cfg.volRatioMax}), efficiency ${eff.toFixed(2)} (min ${cfg.minEfficiency})`,
    });
    if (!volOk) return skip(`Regime: volatility ratio ${volRatio.toFixed(2)} outside band`);
    if (!effOk) return skip(`Regime: market too choppy (efficiency ${eff.toFixed(2)})`);
  } else {
    layers.push({ id: 'regime', name: 'L4 Regime gate', pass: true, skipped: true, detail: 'Disabled' });
  }

  // ---- Layer 2: false-signal filter ---------------------------------------
  // (a) persistence: the same direction must have held for the last N ticks
  let persisted = true;
  for (let k = 1; k < cfg.persistTicks; k++) {
    if (rawSignal(s, last - k, ind) !== sig) { persisted = false; break; }
  }
  // (b) strength: EMA gap must be meaningful relative to normal tick noise
  const noise = avgAbsMove(prices, last, 50);
  const strength = noise > 0 ? Math.abs(s.emaS[last] - s.emaL[last]) / noise : 0;
  const strongEnough = strength >= cfg.minStrength;

  // (c) replay the same rule over history, score each resolved signal
  const lookback = Math.min(ind.signalLookback, last - H);
  const startIdx = Math.max(minData - 5, last - H - lookback);
  // Walk-forward split: the older half "finds" the edge, the recent half must confirm it
  // out-of-sample. A hit-rate that only exists in one half is treated as noise.
  const midIdx = Math.floor((startIdx + (last - H)) / 2);
  let n = 0, wins = 0, nRecent = 0, winsRecent = 0;
  for (let i = startIdx; i <= last - H; i++) {
    const d = rawSignal(s, i, ind);
    if (d === 0) continue;
    const won = (prices[i + H] - prices[i]) * d > 0;
    n++; if (won) wins++;
    if (i > midIdx) { nRecent++; if (won) winsRecent++; }
  }
  metrics.samples = n;
  const pHat = n > 0 ? wins / n : null;
  const falseRate = pHat === null ? 1 : 1 - pHat;
  const enoughData = n >= cfg.minSamples && nRecent >= Math.ceil(cfg.minSamples / 3);
  const pRecent = nRecent > 0 ? winsRecent / nRecent : 0;
  const oosOk = pRecent >= breakeven;

  const l2pass = persisted && strongEnough && enoughData && oosOk && falseRate <= cfg.maxFalseRate;
  layers.push({
    id: 'falseSignal', name: 'L2 False-signal filter', pass: l2pass,
    detail: `held ${cfg.persistTicks}t: ${persisted ? 'yes' : 'no'}, strength ${strength.toFixed(2)} (min ${cfg.minStrength}), ` +
      `${n}/${cfg.minSamples} samples, false-rate ${pHat === null ? 'n/a' : pct(falseRate)} (max ${pct(cfg.maxFalseRate)}), ` +
      `out-of-sample ${nRecent > 0 ? pct(pRecent) : 'n/a'} (needs >= ${pct(breakeven)})`,
  });
  if (!persisted) return skip('False-signal filter: signal just flipped');
  if (!strongEnough) return skip('False-signal filter: EMA gap too weak vs noise');
  if (!enoughData) return skip(`False-signal filter: only ${n}/${cfg.minSamples} past signals`);
  if (!oosOk) return skip(`False-signal filter: recent out-of-sample hit-rate ${pct(pRecent)} below break-even`);
  if (falseRate > cfg.maxFalseRate) return skip(`False-signal filter: ${pct(falseRate)} of past signals failed`);
  metrics.pHat = pHat;

  // ---- Layer 1: EP-EL value filter ------------------------------------------
  const p = pHat as number;
  const ev = p * b - (1 - p);
  metrics.ev = ev;
  const l1pass = ev >= cfg.minEv;
  layers.push({
    id: 'value', name: 'L1 Value filter (EP-EL)', pass: l1pass,
    detail: `p=${pct(p)} vs break-even ${pct(breakeven)}, EV ${ev >= 0 ? '+' : ''}${(ev * 100).toFixed(1)}c per $1 (min ${(cfg.minEv * 100).toFixed(1)}c)`,
  });
  if (!l1pass) return skip(`Value filter: EV ${(ev * 100).toFixed(1)}c below threshold`);

  // ---- Layer 3: fractional Kelly sizing ---------------------------------------
  // Sized on the CONSERVATIVE lower bound of the hit-rate. Overlapping outcomes
  // (a new signal almost every tick) are not independent, so the effective sample
  // is deflated by the horizon before computing the confidence interval.
  const nEff = Math.max(1, n / H);
  const pLower = wilsonLower(p, nEff, cfg.kellyZ);
  metrics.pLower = pLower;
  const fullKelly = (pLower * b - (1 - pLower)) / b;
  metrics.kelly = fullKelly;

  if (cfg.stakingMode !== 'kelly') {
    layers.push({
      id: 'kelly', name: 'L3 Fractional Kelly', pass: true, skipped: true,
      detail: `Staking mode is ${cfg.stakingMode}; Kelly not applied (conservative edge ${pct(pLower)})`,
    });
    return { action: 'trade', direction, stake: ctx.fallbackStake, reason: 'Passed L4/L2/L1', layers, metrics };
  }

  if (fullKelly <= 0) {
    layers.push({
      id: 'kelly', name: 'L3 Fractional Kelly', pass: false,
      detail: `Conservative p ${pct(pLower)} <= break-even ${pct(breakeven)}: no demonstrated edge, stake 0`,
    });
    return skip('Kelly: no statistically demonstrated edge');
  }
  const cap = (cfg.maxStakePct / 100) * ctx.bankroll;
  let stake = ctx.bankroll * fullKelly * cfg.kellyFraction;
  stake = Math.min(stake, cap);
  stake = Math.round(stake * 100) / 100;
  if (stake < cfg.minStake) {
    layers.push({
      id: 'kelly', name: 'L3 Fractional Kelly', pass: false,
      detail: `Kelly stake $${stake.toFixed(2)} below minimum $${cfg.minStake.toFixed(2)}`,
    });
    return skip('Kelly: stake below broker minimum');
  }
  layers.push({
    id: 'kelly', name: 'L3 Fractional Kelly', pass: true,
    detail: `conservative p ${pct(pLower)}, full Kelly ${pct(fullKelly)} x ${cfg.kellyFraction} = $${stake.toFixed(2)} (cap ${cfg.maxStakePct}% of $${ctx.bankroll.toFixed(2)})`,
  });
  return { action: 'trade', direction, stake, reason: 'Passed all layers', layers, metrics };
}
