# Agreement-gated fusion of a vision foundation model and a multimodal LLM for wheat disease diagnosis

Code, frozen prompt and cached model outputs for the manuscript

> Jia, P., & Zhang, P. *Agreement-gated fusion of a vision foundation model and a multimodal large language model for reliable wheat disease diagnosis across image sources.* Under review.

The repository lets you recompute every number, table and statistical figure of the paper from cached outputs, without
re-training the vision model or querying the multimodal model again.

## What is included

| Path | Content |
|---|---|
| `PROTOCOL.md` | Public summary of the time-stamped study protocol: data roles, decision rule, prespecified criteria and outcomes |
| `research/MAINLINE_V2_PROMPT_2026-09-29.txt` | Frozen prompt of expert G61 (gpt-6.1-sol); SHA-256 starts with `03612a75` |
| `scripts/` | Training of expert L, querying of G61 / RAG, decision rule, external evaluations, fusion analyses, near-duplicate audit |
| `outputs/` | Cached outputs: L probabilities (`evaluation.npz`, `L_dino.npy`), G61 JSON responses (`p2_g61/responses/`, keyed by the image hashes in `prepared.json`), RAG responses (`rag*.json`), one-shot result records of the three external evaluations, fold plan, image manifests (identifiers, labels, sources), DINOv3 embeddings used for the audit (`leak_audit/feats.npy`) |
| `manuscript/precision_agriculture_2026/analysis/` | `ms_analysis.py`, `ms_revision.py`: all numbers in the paper (`ms_results.json`, `ms_revision.json`); near-duplicate threshold check |
| `manuscript/precision_agriculture_2026/figures/`, `tables/`, `supplementary/` | Figure and table builders, source data (CSV), exported figures (SVG/PNG), supplementary tables, prompt and schema |

Images are **not** redistributed. They are available from the original publishers (WFD2020, PlantWild v2, MSWDD2022,
Zenodo 13137587 and 15621359, Roboflow Universe, iNaturalist, CerealConv, Mendeley Data, Kaggle; see the paper for
citations and licences). Local paths in the manifests were replaced by `<DATA_ROOT>/`.

## Reproducing the results

```bash
pip install -r requirements.txt
python manuscript/precision_agriculture_2026/analysis/ms_analysis.py
python manuscript/precision_agriculture_2026/analysis/ms_revision.py
python manuscript/precision_agriculture_2026/tables/make_tables.py
python manuscript/precision_agriculture_2026/figures/make_figures.py
```

These steps use only cached outputs and reproduce `ms_results.json`, `ms_revision.json`, the tables and the source data of
Figs. 2, 3, 5, 6 and S1 byte for byte. Figs. 1 and 4 (which show example photographs), the visual threshold check, training
and re-running the experts require the original images; re-querying G61 requires access to the model through the OpenAI Codex CLI (version 0.160.1).

## Decision rule (frozen configuration)

An image receives an automatic answer only when the fine-tuned DINOv3 ensemble (L) and the multimodal model (G61) agree:
"healthy" if L's top class is healthy (p ≥ 0.5) and G61 says healthy; otherwise the disease group (rust, powdery mildew,
septoria) if L's top group score is ≥ 0.5 and G61 names the same group; a rust type only if both name the same rust and G61's
confidence is ≥ 95. All other images are referred for review. See `scripts/paper_p3_hrme.py` (`hrme_v3`).

Released code is the code used in the study. Three modules are public subsets of internal files (image preparation in
`scripts/mainline_v2_vlm.py`, file helpers in `scripts/system_native_tile_experiment.py`, the disabled Codex features in
`src/precall_codex_command.py`); some development options in the scripts refer to outputs of variants that are not part of
the paper and are not included.

## Licence

Code: MIT. Cached model outputs, manifests and figures: CC BY 4.0. Third-party images and data remain under their original licences.

## Use of AI tools

gpt-6.1-sol (OpenAI) was an experimental diagnostic expert in this study. Claude Code (Anthropic, Claude Opus 5.5) assisted with
writing analysis and figure code and with organising the manuscript; the authors checked and take responsibility for all content.
