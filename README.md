# Response–Annotation Mismatch Mining for Ultra-High-Resolution Defect Segmentation

> An open research and engineering idea for finding useful training patches in ultra-high-resolution industrial images by comparing model response maps with imperfect annotations.

## Motivation

Industrial defect inspection often operates on images whose width or height exceeds 10,000 pixels. Defects may be extremely small, thin, fragmented, or low-contrast, while the background contains many structures that resemble defects.

This creates several practical problems:

- Full-resolution dense segmentation is computationally expensive.
- Uniform tiling wastes most computation on easy background.
- Small patches preserve detail but lose context.
- Large patches preserve context but suppress tiny defects.
- Class imbalance is extreme.
- Pixel-accurate annotations are often unavailable or unreliable.
- Conventional hard-example mining usually reduces all errors to a single loss value and does not distinguish failure modes.

The central idea of this repository is:

> Let a coarse generator produce a response map, compare that response map with the available annotation, and mine patches according to the type of disagreement.

The mined regions can then be used to train a more accurate local segmentation model, a verifier, or both.

## Core Pipeline

```text
Ultra-high-resolution image
        |
        v
Cheap coarse generator
        |
        v
Response map R(x, y)
        |
        +----------------------+
        | compare with GT G    |
        v                      v
Mismatch maps            candidate regions
        |                      |
        +----------+-----------+
                   v
        Multi-scale native-resolution crops
                   |
                   v
     Local segmentation / verification model
```

The generator does not need to produce a final segmentation. Its main purpose is to allocate computation by indicating where the model is uncertain, wrong, or easily confused.

## Mismatch Categories

Let:

- `R(x, y) ∈ [0, 1]` be the generator response map.
- `G(x, y) ∈ {0, 1}` be the available defect annotation.
- `P` be a candidate patch.

### 1. High-response background

The model responds strongly to background structures.

```math
S_{FP}(P) = \frac{1}{|P|}\sum_{(x,y)\in P}(1-G(x,y))R(x,y)
```

For thin pseudo-defects, the mean alone may dilute the response. Practical scoring should combine:

```math
S_{FP}^{*}(P)=a\,\operatorname{mean}(R_{bg})
+b\,Q_{0.95}(R_{bg})
+c\,\max(R_{bg})
```

These patches form a `hard_negative_pool`.

### 2. Low-response defect

The annotation contains a defect, but the generator response is weak.

```math
S_{FN}(P)=\frac{1}{|G_P|}\sum_{(x,y)\in P}G(x,y)(1-R(x,y))
```

These patches form a `missed_positive_pool`.

### 3. Low-margin or ambiguous defect

The generator responds to the defect, but not significantly more than to the nearby background.

Let `ring(G)` be a local ring around the annotated defect. Then:

```math
M(P)=\mu_{defect}(R)-\mu_{ring}(R)
```

A small margin means the model sees some signal but cannot distinguish the defect from its surroundings.

These patches form an `ambiguous_positive_pool`.

This category is intentionally different from low-response defects:

- Low-response defect: the model barely sees the defect.
- Low-margin defect: the model reacts, but reacts similarly to nearby background.

### 4. Incomplete defect coverage

A strict shape-mismatch category is unreliable when annotations are imprecise. The relaxed objective should therefore be:

> Prefer complete defect coverage and continuity; tolerate moderate over-segmentation near the annotation.

Instead of requiring exact boundary agreement, only mine cases where the defect is clearly under-covered, fragmented, or interrupted.

#### Tolerant annotation region

Dilate the annotation:

```math
G^{+r}=\operatorname{Dilate}(G,r)
```

Prediction inside `G^{+r}` is tolerated. Only responses far outside this region should be treated as strong false positives.

#### Coverage score

```math
C_{mask}(P)=\frac{|G\cap P^{+r}|}{|G|}
```

For slender defects, a skeleton-based score is often more meaningful:

```math
C_{skel}(P)=\frac{|K_G\cap P^{+r}|}{|K_G|}
```

where `K_G` is the skeleton of the annotation.

A patch becomes a structural hard case only when:

- mask or skeleton coverage is below a relaxed threshold;
- a long continuous segment of the annotated defect receives weak response;
- the response is clearly fragmented;
- an endpoint or major branch is missing.

Moderately thick predictions, small boundary shifts, and local halos should not be treated as major errors.

These patches form an `incomplete_coverage_pool`.

## Error Maps

A simple asymmetric mismatch map is:

```math
E(x,y)=\lambda_{FP}(1-G)R+\lambda_{FN}G(1-R)
```

For recall-oriented industrial inspection, typically:

```math
\lambda_{FN} > \lambda_{FP}
```

A separate ambiguity map can be constructed from the local response margin.

A distance-tolerant false-positive penalty can be defined as:

```math
E_{farFP}(x,y)=R(x,y)(1-G(x,y))w(d(x,G))
```

with:

```math
w(d)=
\begin{cases}
0, & d\le r_1 \\
\frac{d-r_1}{r_2-r_1}, & r_1<d<r_2 \\
1, & d\ge r_2
\end{cases}
```

This avoids punishing small, harmless expansions around noisy annotations.

## Region Extraction

Candidate patches can be generated from mismatch maps using:

1. Thresholding or local peak detection.
2. Connected-component extraction.
3. Component expansion to include context.
4. Multi-scale crop generation.
5. Deduplication of highly overlapping crops.
6. Optional orientation-aware crops for long thin structures.

Recommended paired views for each candidate center:

- Native-resolution detail crop, e.g. `512 × 512`.
- Larger context crop, e.g. `2048 × 2048`, resized to a smaller tensor.

The two views can be fused late in the local model.

## Dynamic Sample Pools

Do not collapse all hard samples into one scalar-ranked list. Different mismatch types represent different optimization objectives.

Suggested pools:

```text
normal_positive_pool
missed_positive_pool
ambiguous_positive_pool
incomplete_coverage_pool
hard_negative_pool
random_background_pool
```

An example training mixture is:

```text
normal positives             25%
missed positives             20%
ambiguous positives          15%
incomplete coverage          15%
hard negatives               20%
random background             5%
```

The exact ratio should be adapted to defect prevalence and operational cost.

## Training Stability

Hard mining should not begin from a random or unstable model.

Recommended procedure:

1. Warm up on ordinary positive and random negative patches.
2. Use an EMA teacher or a previous stable checkpoint to generate response maps.
3. Refresh only part of each hard pool per mining round.
4. Limit how often the same physical defect can be sampled.
5. Keep a non-zero fraction of ordinary and random samples.
6. Track pool-specific metrics rather than only aggregate loss.

Without these controls, the dataset may collapse toward model-specific artifacts.

## Suggested Model Roles

### Coarse generator

Goal: very high recall at low computational cost.

Possible inputs:

- downsampled full image;
- sparse coarse tiles;
- grayscale plus gradient or high-frequency channels.

The generator is allowed to over-respond. Its job is to find regions worth inspecting.

### Local segmentation model

Goal: accurate native-resolution defect segmentation inside selected ROIs.

Potential components:

- lightweight encoder-decoder;
- directional strip convolutions;
- gradient or high-frequency auxiliary channels;
- boundary or centerline auxiliary head;
- connectivity-aware loss;
- detail and context dual branches.

### Hard-negative verifier

Goal: reject pseudo-defects that remain confusing after local segmentation.

Possible inputs:

- tight crop;
- expanded context crop;
- predicted mask;
- gradient features;
- geometric descriptors;
- connected-component statistics.

## Evaluation Philosophy

For safety-critical or high-cost inspection, the metric priority may be:

```text
complete defect coverage
    > continuity
    > local positioning
    > boundary precision
```

Recommended metrics:

- defect-level recall;
- component-level recall;
- skeleton coverage;
- longest missed segment;
- tolerated false-positive area outside a dilated GT region;
- false positives per megapixel;
- inference cost per full image;
- candidate recall before local refinement.

IoU and Dice may still be reported, but they should not be the sole decision metrics for very thin defects with noisy labels.

## Open Research Questions

- How should the tolerance radius adapt to annotation quality and defect width?
- Can the response-margin criterion be calibrated across different materials and imaging conditions?
- How should candidate budgets be allocated when an image contains many pseudo-defects?
- Can patch size, aspect ratio, overlap, and rotation be learned jointly?
- What is the best deduplication strategy for long connected defects spanning multiple crops?
- How can uncertainty estimation improve mismatch mining?
- How should repeated hard samples be scheduled to avoid overfitting?
- Can weak labels, scribbles, or coarse polygons replace precise masks?
- How should defect-level business cost be incorporated into mining and sampling?

## Minimal Experimental Roadmap

### Stage 1: Baseline

- Uniform patch sampling.
- Standard binary segmentation.
- Report patch-level Dice, defect recall, and false positives per megapixel.

### Stage 2: Mismatch mining

- Train a coarse generator.
- Implement the first three mismatch categories.
- Compare mined-patch training with uniform sampling.

### Stage 3: Tolerant structural mining

- Add annotation dilation.
- Add skeleton coverage and longest missed segment.
- Avoid strict boundary-shape penalties.

### Stage 4: Multi-view refinement

- Add native detail and resized context crops.
- Add a verifier for persistent hard negatives.

### Stage 5: Full-image scheduling

- Measure candidate recall, total crop count, memory, and latency.
- Optimize the tradeoff between inspection cost and missed defects.

## Non-Goals

This idea does not assume:

- perfectly accurate pixel-level annotation;
- a specific neural network architecture;
- a specific industrial material;
- that the generator output is a final segmentation;
- that exact boundary alignment is always desirable.

## Contributions

Contributions are welcome in the form of:

- implementations;
- synthetic or public benchmark datasets;
- alternative mismatch scores;
- annotation-noise studies;
- patch scheduling algorithms;
- lightweight model designs;
- industrial case studies;
- negative results.

Please avoid publishing confidential production images or proprietary annotations.

## License

This repository is proposed under the MIT License so that the idea can be freely explored, implemented, modified, and reused.

## Status

Concept proposal. No official implementation is currently maintained.
