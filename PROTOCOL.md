# Study protocol (public summary)

English summary of the internal, time-stamped study protocol (written in Chinese, 7–8 October 2026). Section numbers follow
the internal document, so that references such as "protocol section 14.2" in the code can be looked up here. Every rule below
was written before the computation it governs; the date in each heading is the date it was written. Revisions made after
results had been seen are marked as such. Outcomes are given in brief; all numbers in the paper are recomputed from the
cached outputs by `manuscript/precision_agriculture_2026/analysis/`. The full original document is available from the
corresponding author on request.

Abbreviations: L, fine-tuned DINOv3-B ensemble (3 seeds × 3 source folds); G61, gpt-6.1-sol with the frozen prompt
(`research/MAINLINE_V2_PROMPT_2026-09-29.txt`, SHA-256 `03612a75…`); RAG, G61 given nine labelled reference images retrieved
from the training sources; EVAL450, development set (450 images, 8 WFD2020 sources, 3 source-held folds).

## 1. Data and folds (7 Oct)
EVAL450 with three source-held folds; six classes (healthy, leaf rust, powdery mildew, septoria, stem rust, yellow/stripe rust);
multi-label ground truth. Training data of L per fold: fit sources + public additions + iNaturalist training images.

## 2–3. Experts and expert selection (7 Oct)
Local expert: DINOv3-B (an alternative wheat-specific backbone was compared and not selected; rule: adopt it only if forced-choice
accuracy on EVAL450 exceeds DINOv3 by ≥ 10 images). Multimodal expert: gpt-6.1-sol, queried through the Codex CLI with reasoning
effort "medium", all tools disabled, one fresh session per image, opaque file names; an earlier model version was compared with the
same rule and not selected. Unparsable or failed answers count as no answer.

## 4. Hierarchical decision rule (7 Oct)
Groups: healthy; rust (leaf, stem, yellow rust); powdery mildew; septoria. Levels: healthy / diseased → disease group → specific
disease → referred. Thresholds τh, τd, τg are chosen for each outer fold on the other two source folds only, maximising useful
coverage subject to error ≤ α (α ∈ {5 %, 10 %}). Errors: wrong health status, group not among true groups, disease not among
true labels.

## 5. Metrics and primary criterion (7 Oct)
Useful coverage = share of images answered with "healthy", a disease group or a specific disease; error rate among answers with
one-sided Clopper–Pearson 95 % upper limit. Baselines: L alone and G alone with the same nested threshold selection (wider grids).
Primary criterion at α = 5 %: coverage gain ≥ 10 points over the best single model and upper error limit ≤ 10 %.

## 6. Fresh iNaturalist test (7 Oct; replaced by section 9)

## 6b. IARI laboratory images (7 Oct, before any computation on IARI)
852 detached-leaf laboratory images (an internal collection used earlier for another system) with thresholds frozen on the full
EVAL450. Outcome: the MLLM-free variant gave no automatic answers (L's outputs collapse on black-background laboratory images);
not analysed further.

## 7. Development results (7 Oct)
The first rule version (rust type released whenever L and G agree) did not meet the 5 % error target in nested cross-validation.

## 8. Rule revision v2 (7 Oct; written after seeing the v1 results, disclosed as post hoc)
Optional rust-type level: a rust type is released only if L and G61 name the same rust and G61's confidence ≥ τv
(τv ∈ {never, 0, 70, 80, 90, 95}), selected with the same nested source-held procedure. Development results are not confirmatory.

## 9. Fresh iNaturalist test, final rules (7 Oct, before download and inference)
iNaturalist observations of wheat pathogen taxa (Puccinia triticina and Zymoseptoria tritici always; P. striiformis, P. recondita,
P. graminis and Blumeria graminis only with wheat documented as host); research grade or needs ID; first photo of each observation.
Images with DINOv3 cosine similarity ≥ 0.90 to any local training or development image are removed; one image per observer and day.
Frozen rule evaluated once.

## 10. Strict metric and rule v3 (7 Oct, development, before the iNaturalist evaluation)
Strict metric (primary from here on): "diseased" without a group counts as unanswered. Rule v3 = v2 + two optional agreement gates,
each with an "off" option in the nested selection: health gate (G61 must also say healthy) and group gate (G61's primary label must
fall in the same group). Configuration selected on the full EVAL450 at α = 5 % and frozen: τh = τd = τg = 0.5, rust-type confidence
≥ 95, both gates on.

## 11. Near-duplicate audit of public collections (7 Oct)
Pretrained DINOv3-B embeddings; cosine ≥ 0.90 counted as the same photograph; share of each collection found in any other collection
(`outputs/paper_hrme/leak_audit/audit.json`).

## 12. iNaturalist result (7 Oct, one evaluation)
n = 68. Frozen configuration: coverage 48.5 % (33/68), 3 errors (9.1 %, upper limit 21.9 %); all baselines answered 0 images.
Coverage criterion met, error-limit criterion not met. A second configuration selected at α = 10 % (health gate off) made 9 errors
among 39 answers and was not used further.

## 13. External test set ETS (7 Oct, before any ETS inference)
### 13.1 Construction
Independent sources, single-label images only: PlantWild v2 (wheat), MSWDD2022, Zenodo 13137587 (camera originals), Roboflow
new-wheat-disease v2, stem-rust-v1 and Puccinia-triticina v7 (images with a single annotated class). Removed: images with cosine
≥ 0.90 to any local training or development image; within ETS one image per connected group at ≥ 0.90. At most 40 images per
(source, class), seed 20261007.
### 13.2 Planned revision
A further revision of the rule was planned here and superseded by section 14 before any ETS computation; it was not carried out.

## 14. Retrieval-augmented MLLM (7 Oct, before results)
For each image, nine labelled reference images (two per rust, one each for healthy, powdery mildew and septoria) are retrieved by
pretrained DINOv3 similarity from the training sources of the image's fold; candidates with cosine ≥ 0.90 to the query are removed.
Pilot on 90 EVAL450 images (seed 20261008): continue only if RAG beats plain G61 by ≥ 6 images (outcome: 64 vs 55, continued).
### 14.2 Full development run and ETS rule (7 Oct, before any ETS metric)
Fusion selected on EVAL450 only and frozen: RAG answer, with L's top class when RAG is uncertain. Primary criterion on ETS (forced
choice): ≥ 5 points over the best single expert with paired-bootstrap 95 % CI lower limit > 0. Secondary: agreement-gated system
(section 10 configuration) on ETS.

## 15. ETS result (8 Oct, one evaluation)
Primary criterion not met: −0.4 points [−2.4, 1.8] vs G61 with L fallback. Agreement-gated system (secondary): coverage 72.6 %,
error 1.0 %.

## 16. Adaptive retrieval (8 Oct, before computation; ETS values post hoc)
Candidate rules A0–A4 (A3: RAG only when G61 is uncertain or disagrees with L; L when RAG is uncertain), selected on EVAL450.

## 17. Second external test set ETS2 and confirmatory hypotheses (8 Oct, before download and inference)
Zenodo 15621359 (Henan field images; single-class images: powdery mildew, leaf rust, stripe rust, healthy leaf) and PlantSeg v7
wheat images. Removed: cosine ≥ 0.90 to local data or ETS; one per connected group. At most 60 images per (source, class), seed 20261008.
H1: A3 ≥ G61 with L fallback + 3 points (CI lower limit > 0). H2: G61 with L fallback ≥ L + 3 points (CI lower limit > 0).
H3: agreement-gated system error ≤ 5 %, upper limit ≤ 8 %, coverage ≥ 60 %. Evaluated once.

## 18. ETS2 result (8 Oct)
H1 not met (+1.3 points [−1.3, 3.8]); H2 not met (−18.3 points [−23.8, −12.9]; late stripe rust read as septoria by the MLLM);
H3 met (coverage 68.3 %, error 1/164, upper limit 2.9 %).

## 19. Learned arbiters (8 Oct, before modelling; retrospective use of already examined data)
Pooled 1,310 images from 16 sources; leave-one-source-out cross-validation. Features: L probabilities, G61 label, confidence,
image type, assessability and seven symptom fields, RAG label and confidence. Models fixed in advance: multinomial logistic
regression (standardised, C = 1, balanced) and gradient-boosted trees (defaults, 200 iterations). Criterion: ≥ 5 points over the
best fixed rule (CI lower limit > 0) and no collection > 2 points worse. Outcome: not met (−8.7 points).

## 20. Knowledge-rule arbiter (8 Oct, before computation)
When L and G61 disagree, the candidate whose defining symptoms (from the frozen prompt) are better supported by G61's own symptom
report is chosen; ties go to RAG if it names one of the candidates, otherwise to the majority vote. Same criterion as section 19.
Outcome: not met (−1.9 points vs majority vote).

## 21. Local rule selection with K labelled images (8 Oct, before computation)
K ∈ {5, 10, 20} labelled images per source (50 random draws) choose among L, G61 with L fallback, RAG with L fallback and the
majority vote; evaluated on the remaining images; sources with fewer than 2K images use the vote. Criterion (K = 10): ≥ 3 points
over the vote and no collection > 1 point worse. Outcome: not met (+2.3 points; worst collection −1.4 points).
