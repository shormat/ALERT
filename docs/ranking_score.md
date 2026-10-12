# Frame-level Piérard ranking score $R_I$

## Why a frame-level score in addition to $S_{\mathrm{cont}}$

The continuous score reported in Section 5 / Table 6 of the report,
$S_{\mathrm{cont}} = \sum_k \tilde w_k m_k$, is a weighted average of
three aggregate metrics: $\mathrm{mAP}_{50}$ (image-level detection
eval, 10 K BDD100K still images), $\mathrm{IDF1}$ (sequence-level MOT
eval, 100 videos at 5 fps GT) and $\mathrm{mIoU}$ (pixel-level
drivable-area eval). Those three metrics live in three distinct
random experiments with three distinct sample spaces, so pooling them
violates the single-random-experiment assumption underlying the
canonical ranking score from Lecture 6 (ELEN0449, S. Piérard). The
weights $w_k$ are application-dependent preferences, not probabilities
on a common $\Omega$, and the cross-experiment sum is not invariant
under reasonable changes of the importance schedule.

To complement $S_{\mathrm{cont}}$ — without replacing it — we add a
frame-level ranking score $R_I$ defined on a single random experiment:
sample one labelled frame uniformly from the MOT val set, observe the
pipeline's joint output, score its satisfaction, and weight by the
importance attached to that frame.

## Formula

For one random experiment $(\Omega, P)$ with outcome-level
satisfaction $S\colon\Omega\to[0,1]$ and importance
$I\colon\Omega\to\mathbb{R}_{\ge 0}$,

$$
R_I(P) \;=\; \frac{\sum_{\omega\in\Omega} S(\omega)\,I(\omega)\,P(\{\omega\})}
                  {\sum_{\omega\in\Omega} I(\omega)\,P(\{\omega\})}.
$$

Our experiment samples one of $N$ MOT val labelled frames uniformly:

$$P(\{\omega\}) = \tfrac{1}{N}\quad\forall\omega.$$

The factor $1/N$ cancels in numerator and denominator, leaving the
finite-sample estimator we actually compute:

$$
R_I \;=\; \frac{\sum_{\text{frames}} S(\text{frame})\,I(\text{frame})}
               {\sum_{\text{frames}} I(\text{frame})}.
$$

`compute_R_I` in `alert.evaluation.ranking_score` keeps the
P-aware form (`probabilities=...`) so any future non-uniform sampling
scheme — day/night rebalancing, sequence-level pre-sampling,
importance sampling for rare critical scenes — can be plugged in
without touching downstream code.

## Outcome and the three signals on a frame

An outcome is $\omega = (\text{frame}, \text{GT}, \text{predictions})$.
For one labelled MOT frame we extract three independent satisfaction
signals and combine them.

* $S_\mathrm{det}(\omega) = 1$ iff the per-frame macro F1 over the seven
  movable BDD100K MOT classes (`pedestrian, rider, car, truck, bus,
  motorcycle, bicycle`) at IoU 0.5 is $\geq \tau_\mathrm{det} = 0.5$.
  Frames with no GT and no pred among those classes contribute
  $S_\mathrm{det} = 1$ (vacuous "do nothing" agreement).
* $S_\mathrm{seg}(\omega) = 1$ iff the direct-drivable IoU on that
  frame is $\geq \tau_\mathrm{seg} = 0.7$.
* $S_\mathrm{trk}(\omega) = 1$ iff no VRU GT track on that frame is
  matched to a pred id different from the one it last matched within
  the sequence. VRU set: `{pedestrian, rider, motorcycle, bicycle}`.
  Unmatched VRU GTs (FN) do not count as switches; they are picked up
  by $S_\mathrm{det}$.
* $S(\omega) = S_\mathrm{det}\wedge S_\mathrm{seg}\wedge S_\mathrm{trk}$.

We also report the three sub-task $R_I$ scores individually so it is
visible which task drags the joint score down.

## Importance profiles

Three profiles defined on a labelled frame:

| Profile | $I(\omega)$ |
|---|---|
| uniform | $1$ |
| risk-weighted | $5$ if any VRU GT footprint overlaps the GT direct-drivable mask by $\geq 30\%$, else $1$ |
| VRU-present | $1$ if the frame has any VRU GT, else $0$ |

The VRU footprint is the same thin band below the bbox bottom edge
used by $p_i$ in Section 4.5 (`risk_reasoner._in_path_likelihood`):
`band_h = max(4, (iy2 - iy1) // 6)` sampled in
`[iy2 - 2 : iy2 + band_h, ix1 : ix2]`. This keeps the importance
profile geometrically consistent with the risk reasoner.

The VRU-present profile reduces $R_I$ to a conditional —
$P(S = 1 \mid \text{VRU present})$ — exactly the slide 66 read.

## Ground-truth alignment

BDD100K detection GT lives on still keyframes (one per clip), drivable
GT lives on those same keyframes, and MOT GT lives on labelled video
frames at 5 fps (~200 per clip). The three sub-satisfactions need to
be defined on the same frame set. Three options were considered:

1. Strict intersection (the keyframe alone): tiny, and the keyframe is
   in the segmentation training set — leakage.
2. Propagate the GT direct-drivable mask from the keyframe to every
   labelled MOT frame in the same clip (assumes drivable area is
   roughly constant across a 40 s urban clip).
3. Decouple: report $R_I$ for $S_\mathrm{det} \wedge S_\mathrm{trk}$
   on MOT frames, plus $R_I$ for $S_\mathrm{seg}$ alone on the 10 K
   detection val keyframes.

We use option 2. Rationale: it is the only option that preserves the
joint-task semantics that motivate replacing $S_\mathrm{cont}$.
Option 3 would collapse to "two more cross-experiment scores", which
is exactly the canonicality problem $R_I$ was meant to fix.

### Caveat — segmentation GT lives in the seg training set

All 1 000 BDD100K MOT clips are sourced from the BDD100K official
train split. The single still keyframe of each MOT-val clip therefore
lives in the alvaromalfaro train set, which is the segmentation
training set. Absolute $S_\mathrm{seg}$ values are optimistic. The
ranking is still informative because every pipeline draws its
segmentation predictions from the same training-set distribution, so
the bias is uniform across the 5 ranked configurations. We document
this in the report's discussion of $R_I$ and recommend that future
work train the segmenter on a stem set disjoint from the MOT split.

## Outputs

`scripts/eval_ranking_score.py` writes two artefacts per pipeline:

* `outputs/eval/ranking_<det>_<seg>_<trk>_<split>.json` — the
  aggregate $R_I$ block (joint + sub-task, per profile, plus support
  counts) plus a `meta` block recording every flag value. This is the
  source of truth for the LaTeX fragment.
* `outputs/eval/ranking_<det>_<seg>_<trk>_<split>.jsonl` — one record
  per labelled frame with `(s_det, s_seg, s_trk, s_joint, I_*)`. This
  is what makes ablations cheap: changing $\tau_\mathrm{det}$,
  $\tau_\mathrm{seg}$, the VRU set or the importance profile is a
  pure post-process — no re-inference.

`scripts/eval_ranking_score_batch.py` runs (or re-uses cached) results
for the 5 canonical pipelines from Table 6 and emits the LaTeX table
fragment + a rank-reversal annotation comparing $R_I$ orderings to the
$S_{\mathrm{cont}}$ Balanced ordering.

## Cost

Inference runs every raw video frame at 30 fps through the detector
and tracker, but only every $\sim$6th raw frame through the
segmentor (only labelled MOT indices need a seg prediction). On a
single GPU this is $\sim$30–60 min per pipeline for the 100 val clips,
i.e. 3–5 h to evaluate all five rows of Table 6.
