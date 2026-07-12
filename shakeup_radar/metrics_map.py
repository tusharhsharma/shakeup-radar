"""Metric-family mapping — decision cache item #1.

Every entry here is a judgment a fresh rewrite would have to re-derive:
which evaluation metrics behave alike w.r.t. rank stability. Families are
deliberately coarse (sparse strata are worse than coarse ones — decision #2).

Unknown metrics fall to 'other' loudly (listed in the fit report), never
silently — decision #3.
"""

# EvaluationAlgorithmAbbreviation -> family
METRIC_FAMILY = {
    # threshold-free ranking metrics: rank-stable under mild noise
    "AUC": "auc_rank", "GINI": "auc_rank", "ROC": "auc_rank",
    # probabilistic losses: sensitive to calibration, moderate stability
    "LogLoss": "prob_loss", "Log Loss": "prob_loss", "MulticlassLoss": "prob_loss",
    "MCLogLoss": "prob_loss", "CrossEntropy": "prob_loss", "BrierScore": "prob_loss",
    # squared/absolute error regression
    "RMSE": "reg_error", "RMSLE": "reg_error", "MAE": "reg_error", "MSE": "reg_error",
    "R2": "reg_error", "SMAPE": "reg_error", "MAPE": "reg_error", "MedianAE": "reg_error",
    # classification accuracy-like: quantized, tie-prone -> jumpy ranks
    "Accuracy": "accuracy_like", "CategorizationAccuracy": "accuracy_like",
    "F1": "accuracy_like", "MacroF1": "accuracy_like", "MicroF1": "accuracy_like",
    "MCC": "accuracy_like", "QuadraticWeightedKappa": "accuracy_like",
    "Kappa": "accuracy_like", "BalancedAccuracy": "accuracy_like",
    # retrieval / recommendation: top-k truncation amplifies noise
    "MAP@{K}": "retrieval", "MAP": "retrieval", "NDCG": "retrieval",
    "MRR": "retrieval", "AveragePrecision": "retrieval", "Precision@K": "retrieval",
    # segmentation / detection overlap scores
    "Dice": "overlap", "IoU": "overlap", "Jaccard": "overlap", "meanFScore": "overlap",
    "MeanBestErrorAtK": "overlap", "IntersectionOverUnion": "overlap",
    # text/sequence similarity
    "BLEU": "text_sim", "ROUGE": "text_sim", "WordError": "text_sim",
    "LevenshteinDistance": "text_sim", "CRPS": "prob_loss",
}

FAMILIES = sorted(set(METRIC_FAMILY.values())) + ["other"]

# decision #22 (real-data finding, 2026-07-12): Meta Kaggle's abbreviation
# vocabulary is wilder than any explicit table — MCAUC, FScoreBetaMicro,
# MWCRMSE, OpenImagesObjDetectionSegmentationAP... After the exact table,
# substring rules catch the compositional names. Order matters: more
# specific patterns first. Numeric "abbreviations" (recent Meta Kaggle
# stores an ID there) are handled upstream by falling back to the Name.
PATTERN_RULES = [
    (("NDCG", "MAP@", "MRR", "GlobalAP", "DetectionAP", "SegmentationAP",
      "AveragePrecision"), "retrieval"),
    (("AUC", "GINI",), "auc_rank"),
    (("RMSLE", "RMSE", "WMAE", "MAE", "SMAPE", "MAPE", "CRMSE"), "reg_error"),
    (("LOSS", "DEVIANCE", "ENTROPY", "BRIER", "CRPS"), "prob_loss"),
    (("FSCORE", "F1", "ACCURACY", "KAPPA", "MCC", "MATTHEWS"), "accuracy_like"),
    (("LEVENSHTEIN", "WORDERROR", "BLEU", "ROUGE"), "text_sim"),
    (("DICE", "IOU", "JACCARD", "FBETA"), "overlap"),
]

def family_of(abbrev):
    if not isinstance(abbrev, str) or not abbrev.strip():
        return "other"
    a = abbrev.strip()
    if a in METRIC_FAMILY:
        return METRIC_FAMILY[a]
    # normalized retry — decision #4: case/spacing variants exist across years
    key = a.replace(" ", "").lower()
    for k, v in METRIC_FAMILY.items():
        if k.replace(" ", "").lower() == key:
            return v
    up = a.upper()
    for pats, fam in PATTERN_RULES:                       # decision #22
        if any(p.upper() in up for p in pats):
            return fam
    return "other"
