**Table 1 Image collections and their roles in the study**

| Collection | Role | Sources | Images | Labels (images per class) |
|---|---|---|---|---|
| EVAL450 | Development (rule design and threshold selection) | WFD2020, 8 contributing sub-sources (3 source-held folds) | 450 | stripe rust 131, leaf rust 126, stem rust 90, healthy 67, powdery mildew 42, septoria 28 |
| ETS | External test (prespecified, evaluated once) | PlantWild v2 wheat (200), MSWDD2022 (80), wheat powdery mildew camera set (40), Roboflow new-wheat-disease v2 (160), stem-rust v1 (32), Puccinia triticina v7 (40) | 552 | leaf rust 129, powdery mildew 120, stripe rust 120, stem rust 101, healthy 42, septoria 40 |
| ETS2 | External confirmatory test (prespecified hypotheses H1–H3) | Henan field data set, Yuanyang 2023 (smartphone and DSLR) | 240 | healthy 60, leaf rust 60, powdery mildew 60, stripe rust 60 |
| iNat | External test (prespecified, evaluated once) | iNaturalist pathogen observations with wheat as host (48 observers) | 68 | leaf rust 36, septoria 23, powdery mildew 5, stripe rust 4 |

_EVAL450 is multi-label (32 images carry more than one label); class counts are label occurrences._
_All external images were de-duplicated against every training, development and earlier test pool (DINOv3 embedding cosine similarity ≥ 0.90)._
_Training pool of expert L per development fold: 2,616 images (600 WFD2020 images from the sources of the other two folds, 999 CerealConv, 406 Mendeley wheat leaf and 611 iNaturalist wheat images from observations not in the iNat test collection)._

**Table 2 Hypotheses fixed before each evaluation and their outcomes**

| Data | Comparison | Success criterion | Result [95 % CI] | Met |
|---|---|---|---|---|
| iNat | Frozen system vs best single expert, same selection procedure (α = 5 %) | Coverage gain ≥ 10 points; one-sided upper 95 % limit of error ≤ 10 % | Coverage 48.5 % vs 0 %; error 3/33 (9.1 %), upper limit 21.9 % | Coverage: yes; error limit: no |
| ETS | RAG with L fallback vs best single expert (forced choice) | Gain ≥ 5 points and 95 % CI lower limit > 0 | −0.4 points [−2.4, 1.8] vs G61 with L fallback | No |
| ETS2, H1 | Adaptive RAG vs G61 with L fallback (forced choice) | Gain ≥ 3 points and CI lower limit > 0 | +1.3 points [−1.3, 3.8] | No |
| ETS2, H2 | G61 with L fallback vs L (forced choice) | Gain ≥ 3 points and CI lower limit > 0 | −18.3 points [−23.8, −12.9] | No |
| ETS2, H3 | Frozen agreement-gated system | Error ≤ 5 %, one-sided upper 95 % limit ≤ 8 %, coverage ≥ 60 % | Error 1/164 (0.6 %), upper limit 2.9 %, coverage 68.3 % | Yes |
| 16 sources (retrospective) | Learned arbiter (logistic regression, all features) vs best fixed rule | Gain ≥ 5 points, CI lower limit > 0, no collection > 2 points worse | −8.7 points [−10.7, −6.7] | No |
| 16 sources (retrospective) | Knowledge-rule arbiter vs majority vote | As above | −1.9 points [−2.9, −1.0] | No |
| 16 sources (retrospective) | Local rule selection with 10 labelled images vs majority vote | Gain ≥ 3 points, no collection > 1 point worse | +2.3 points; worst collection −1.4 points (iNat) | No |

_iNat, ETS and ETS2 were each evaluated once after the criteria had been time-stamped; the retrospective criteria were fixed before computation but on data whose results had been examined._
_On ETS the frozen system was a secondary, descriptive analysis (coverage 72.6 %, error 1.0 %). A second configuration selected at α = 10 % was also evaluated on iNat (9 errors among 39 answers) and was not used further._

**Table 3 Forced-choice accuracy (%) of single experts, simple combinations and the two-expert selection bound**

| Method | EVAL450 (n = 450) | ETS (n = 552) | ETS2 (n = 240) | iNat (n = 68) |
|---|---|---|---|---|
| Vision model L (DINOv3) | 54.9 (247/450) [50.2–59.6] | 84.6 (467/552) [81.5–87.5] | 95.8 (230/240) [93.3–98.3] | 61.8 (42/68) [50.0–73.5] |
| MLLM G61 alone (uncertain counted as wrong) | 58.4 (263/450) [54.0–63.1] | 81.3 (449/552) [78.1–84.4] | 68.8 (165/240) [62.9–74.6] | 51.5 (35/68) [39.7–63.2] |
| Retrieval-augmented MLLM (RAG) alone | 67.3 (303/450) [62.9–71.8] | 83.0 (458/552) [79.7–86.1] | 71.2 (171/240) [65.4–77.1] | 67.6 (46/68) [55.9–77.9] |
| G61 with L fallback | 67.1 (302/450) [62.9–71.6] | 88.2 (487/552) [85.5–90.8] | 77.5 (186/240) [72.1–82.9] | 64.7 (44/68) [52.9–76.5] |
| RAG with L fallback | 72.9 (328/450) [68.7–76.9] | 87.9 (485/552) [85.1–90.6] | 74.6 (179/240) [68.8–80.0] | 72.1 (49/68) [61.8–82.4] |
| Majority vote of L, G61 and RAG | 71.8 (323/450) [67.6–76.0] | 88.9 (491/552) [86.2–91.5] | 78.8 (189/240) [73.3–83.8] | 70.6 (48/68) [58.8–80.9] |
| Two-expert selection bound (L or G61 correct)a | 72.7 (327/450) [68.7–76.7] | 95.1 (525/552) [93.1–96.9] | 97.5 (234/240) [95.4–99.2] | 70.6 (48/68) [60.3–80.9] |

_Values: accuracy (correct/total) [95 % bootstrap interval], computed on all images of each collection. EVAL450 values are source-held (each image scored by models that never saw its source)._
_a Oracle quantity: share of images on which at least one of L and G61 is correct; not attainable without the true label._
_RAG on iNat was computed after the iNat evaluation and is therefore retrospective for that collection. Accuracy on the subset of images where L and G61 agree is shown in Fig. 3c._

**Table 4 Automatic answers of the frozen agreement-gated system on the external collections**

| Collection | Answered (coverage %) | Errors | Accuracy % [95 % CI] | Specific / group / healthy | Group level %: system / L / G61 | System − L, points [95 % CI] | System − G61, points [95 % CI] |
|---|---|---|---|---|---|---|---|
| ETS | 401/552 (72.6) | 4 | 99.0 [97.5–99.7] | 245 / 148 / 8 | 99.3 / 96.8 / 97.0 | +2.5 [1.0, 4.2] | +2.2 [1.0, 3.8] |
| ETS2 | 164/240 (68.3) | 1 | 99.4 [96.6–100.0] | 77 / 46 / 41 | 100.0 / 99.4 / 83.5 | +0.6 [0.0, 1.9] | +16.5 [12.2, 23.2] |
| iNat | 33/68 (48.5) | 3 | 90.9 [75.7–98.1] | 19 / 14 / 0 | 93.9 / 78.8 / 93.9 | +15.2 [3.2, 31.0] | 0.0 [−9.7, 10.5] |
| Pooled external (8 sources) | 598/860 (69.5) | 8 | 98.7 [97.4–99.4] | 341 / 208 / 49 | – | – | – |

_Columns 2–5: accuracy among answered images at the level at which each answer was given; Clopper–Pearson intervals._
_Columns 6–8: all methods at the same number of answered images, scored at group granularity (healthy, rust, powdery mildew, septoria); single experts answer their most confident images. Difference intervals come from a bootstrap that repeats the selection in each resample._
_Development estimate (source-held nested cross-validation on EVAL450): coverage 66.9 %, error rate 5.0 % (15/301). Per-source results are shown in Fig. 5._

**Table 5 Forced-choice accuracy (%) of fusion strategies in retrospective leave-one-source-out analysis (16 sources, 1,310 images)**

| Strategy | Labelling needed at a new source | All sources | EVAL450 | ETS | ETS2 | iNat |
|---|---|---|---|---|---|---|
| L | None | 75.3 | 54.9 | 84.6 | 95.8 | 61.8 |
| G61 with L fallback | None | 77.8 | 67.1 | 88.2 | 77.5 | 64.7 |
| RAG with L fallback | None | 79.5 | 72.9 | 87.9 | 74.6 | 72.1 |
| Majority vote (L, G61, RAG) | None | 80.2 | 71.8 | 88.9 | 78.8 | 70.6 |
| Learned arbiter, logistic regression, L and G61 features | None (trained on 15 other sources) | 75.4 | 63.1 | 86.1 | 75.8 | 69.1 |
| Learned arbiter, logistic regression, all features | None (trained on 15 other sources) | 71.5 | 55.6 | 84.2 | 74.6 | 63.2 |
| Learned arbiter, gradient boosting, all features | None (trained on 15 other sources) | 71.6 | 54.2 | 83.7 | 78.8 | 63.2 |
| Knowledge-rule arbiter | None | 78.3 | 68.7 | 87.9 | 77.1 | 69.1 |
| Local rule selection, K = 5 | 5 labelled images per source | 81.8 | 71.9 | 88.1 | 89.9 | 68.5 |
| Local rule selection, K = 10 | 10 labelled images per source | 82.5 | 72.1 | 88.1 | 93.0 | 69.2 |
| Local rule selection, K = 20 | 20 labelled images per source | 82.9 | 72.5 | 88.0 | 94.7 | 69.1 |
| Per-source best fixed rule (all labels)b | Oracle reference | 84.9 | 74.7 | 90.0 | 95.8 | 72.1 |
| Two-expert selection bound (L or G61 correct)b | Oracle reference | 86.6 | 72.7 | 95.1 | 97.5 | 70.6 |
| Three-expert bound (L, G61 or RAG correct)b | Oracle reference | 89.2 | 78.4 | 95.8 | 97.9 | 76.5 |

_Retrospective analysis on collections whose results had already been examined; learned components were always fitted on the other 15 sources, using cached expert outputs._
_Local rule selection chooses, per source, the most accurate of four fixed rules (L; G61 with L fallback; RAG with L fallback; majority vote) on K random labelled images and is evaluated on the remaining images (mean of 50 draws; for K = 10 the pooled accuracy of the central 95 % of draws was 78.7–84.4 %). Sources with fewer than 2K images use the majority vote (K = 10: 3 sources, 27 images)._
_b Oracle references that use the true labels. RAG-based rules can exceed the two-expert bound because they draw on a third output; rules that only select among the answers of L, G61 and RAG cannot exceed the three-expert bound._
