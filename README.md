🧠 Adaptive, Self-Explaining Deep Learning System for Early Detection of Cognitive Impairment from Spontaneous Speech

An audio-first, interpretable-by-design screening system for Alzheimer's Dementia (AD). From a single spoken description of the Boston Cookie-Theft picture, it returns a diagnosis, a clinician-readable explanation, and a confidence-based decision on when enough speech has been heard.

Developed at the School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore, India.

✨ Highlights
Intrinsically interpretable: predictions pass through a supervised concept bottleneck of 24 clinical speech markers and a linear head, so every prediction is an exact additive scorecard. No post-hoc SHAP/LIME needed.
Multimodal: acoustic branch (WavLM + per-utterance prosody) and linguistic branch (Whisper ASR → RoBERTa), combined with gated cross-attention.
Adaptive: a confidence-based early-exit decides when enough speech has been observed. About 10 seconds of speech already reaches full-recording performance.
No accuracy penalty for interpretability: matches a black-box counterpart (−0.003 AUC).
Rigorous evaluation: speaker-independent, repeated stratified group cross-validation, with leakage verified programmatically.
📊 Results

Evaluated on the full DementiaBank Pitt Cookie-Theft corpus (552 recordings, 292 speakers; 309 AD / 243 Control) under speaker-independent, 3× repeated stratified group 5-fold CV.

Model	OOF ROC-AUC	F1
Linguistic markers + Logistic Regression	0.665	0.666
All 24 markers + Random Forest	0.716	0.716
Black-box (cross-attention, MLP head)	0.719	0.717
Concept-Bottleneck (proposed)	0.722 ± 0.014	0.695

Full metrics (proposed model):

Metric	Mean ± SD	Metric	Mean ± SD
ROC-AUC	0.722 ± 0.014	Sensitivity	0.669 ± 0.048
PR-AUC	0.738 ± 0.009	Specificity	0.679 ± 0.062
Accuracy	0.673 ± 0.003	Precision	0.729 ± 0.025
Balanced accuracy	0.674 ± 0.008	F1	0.695 ± 0.015
MCC	0.347 ± 0.014		

Other key findings

Concept fidelity: mean r = 0.69 between predicted and true markers; 17 of 24 markers above r = 0.6 (speech rate, shimmer, MLU, HNR, F0 statistics and loudness exceed r = 0.85).
Early exit: accuracy plateaus at roughly 10 s of patient speech.
Balanced benchmark: on the ADReSSo-2021 subset (166 recordings) the same framework reaches ROC-AUC ≈ 0.84.

⚠️ Results across datasets and protocols are not directly comparable. Speaker-independent evaluation on the full, unbalanced, multi-session Pitt corpus is considerably harder than on curated balanced subsets.

## 🏗️ Architecture

![Architecture](Architecture.png)

Components
Stage	Details
Diarisation	Gold transcript alignments during development; VAD / speaker-diarisation model for deployment
Acoustic encoder	WavLM-base-plus (768-d) + 8-d prosody (F0 mean/SD/range, jitter, shimmer, HNR, intensity mean/SD)
Linguistic encoder	Whisper large-v3-turbo transcripts → RoBERTa-base (768-d)
Fusion	Fusion dim 128, 4 attention heads, per-token learned gate
Discourse encoder	BiGRU (hidden 64) with attention pooling
Bottleneck	24 supervised concepts, linear prediction head
Encoders	Frozen (appropriate for corpus size)
🔍 Interpretability
Global importance: longer and more frequent pauses, jitter, pronoun over-use and filled pauses push toward AD; higher lexical diversity, pitch variation and utterance length push toward Control. These directions are consistent with clinical findings on cognitive decline.
Per-case scorecard: each contribution wₖcₖ is exact and sums to the logit. A one-line explanation is generated, for example:

"Predicted Control (89%): few long pauses, low voice jitter, richer sentence structure."

📁 Dataset
Property	Value
Corpus	DementiaBank Pitt (Cookie-Theft task)
Recordings / speakers	552 / 292
Class distribution	309 AD, 243 Control
Audio	Noise-reduced, resampled to 16 kHz mono
Labels and diarisation	Transcript-derived (gold)
Balanced reference subset	ADReSSo-2021 (166 recordings)

The DementiaBank data is not included in this repository. Access requires registration and agreement to the DementiaBank terms: https://dementia.talkbank.org/

⚙️ Implementation Details
Optimiser: Adam (lr = 1e-3, weight decay = 1e-4)
Batch size: 16, up to 80 epochs, early stopping on validation AUC
Evaluation: speaker-grouped, label-stratified 5-fold CV, repeated 3×, with OOF estimates over all 552 recordings
Threshold: tuned on a validation split via Youden's J; ROC-AUC and PR-AUC are the primary threshold-free metrics
Baselines: logistic regression and random forest on the 24 markers, plus a black-box variant (identical model with an MLP head and no bottleneck)
