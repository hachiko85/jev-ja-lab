from __future__ import annotations

import math
import statistics
from typing import Any


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def calibration_metrics(
    probabilities: list[list[float]], gold_indices: list[int], bins: int = 10
) -> dict[str, float]:
    if not probabilities:
        return {}
    epsilon = 1e-12
    log_likelihood = sum(
        math.log(max(row[gold], epsilon))
        for row, gold in zip(probabilities, gold_indices, strict=True)
    )
    nll = -log_likelihood / len(probabilities)
    brier = 0.0
    for row, gold in zip(probabilities, gold_indices, strict=True):
        brier += sum((value - float(index == gold)) ** 2 for index, value in enumerate(row))
    brier /= len(probabilities)
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for row, gold in zip(probabilities, gold_indices, strict=True):
        predicted = max(range(len(row)), key=row.__getitem__)
        confidence = row[predicted]
        bucket = min(int(confidence * bins), bins - 1)
        buckets[bucket].append((confidence, float(predicted == gold)))
    ece = 0.0
    for bucket in buckets:
        if bucket:
            confidence = statistics.fmean(value[0] for value in bucket)
            accuracy = statistics.fmean(value[1] for value in bucket)
            ece += len(bucket) / len(probabilities) * abs(accuracy - confidence)
    return {"nll": nll, "brier_score": brier, "ece": ece}


def binary_classification_metrics(
    predicted_indices: list[int], gold_indices: list[int]
) -> dict[str, float]:
    pairs = list(zip(predicted_indices, gold_indices, strict=True))
    true_positive = sum(p == 1 and g == 1 for p, g in pairs)
    false_positive = sum(p == 1 and g == 0 for p, g in pairs)
    false_negative = sum(p == 0 and g == 1 for p, g in pairs)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _average_ranks(values: list[int]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index, _ in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if not left:
        return 0.0
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum(
        (x - mean_left) * (y - mean_right) for x, y in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((x - mean_left) ** 2 for x in left) * sum((y - mean_right) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def _quadratic_weighted_kappa(predicted: list[int], gold: list[int]) -> float:
    if not predicted:
        return 0.0
    category_count = max([*predicted, *gold]) + 1
    if category_count < 2:
        return 1.0
    observed = [[0.0] * category_count for _ in range(category_count)]
    predicted_hist = [0.0] * category_count
    gold_hist = [0.0] * category_count
    for prediction, target in zip(predicted, gold, strict=True):
        observed[target][prediction] += 1.0
        gold_hist[target] += 1.0
        predicted_hist[prediction] += 1.0
    total = float(len(predicted))
    observed_error = 0.0
    expected_error = 0.0
    scale = float((category_count - 1) ** 2)
    for target in range(category_count):
        for prediction in range(category_count):
            weight = (target - prediction) ** 2 / scale
            observed_error += weight * observed[target][prediction] / total
            expected_error += (
                weight * gold_hist[target] * predicted_hist[prediction] / (total * total)
            )
    return 1.0 - observed_error / expected_error if expected_error else 1.0


def ordinal_metrics(predicted_indices: list[int], gold_indices: list[int]) -> dict[str, float]:
    errors = [
        prediction - gold
        for prediction, gold in zip(predicted_indices, gold_indices, strict=True)
    ]
    if not errors:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "spearman": 0.0,
            "quadratic_weighted_kappa": 0.0,
            "normalized_quadratic_weighted_kappa": 0.5,
        }
    kappa = _quadratic_weighted_kappa(predicted_indices, gold_indices)
    return {
        "mae": statistics.fmean(abs(error) for error in errors),
        "rmse": math.sqrt(statistics.fmean(error**2 for error in errors)),
        "spearman": _pearson(_average_ranks(predicted_indices), _average_ranks(gold_indices)),
        "quadratic_weighted_kappa": kappa,
        "normalized_quadratic_weighted_kappa": max(0.0, min(1.0, (kappa + 1.0) / 2.0)),
    }


def summarize(
    predictions: list[dict[str, Any]], elapsed_seconds: float, task_type: str = "choice"
) -> dict[str, int | float | str]:
    total = len(predictions)
    correct = sum(int(row["predicted_index"] == row["gold_index"]) for row in predictions)
    latencies = [float(row["latency_ms"]) for row in predictions]
    summary: dict[str, int | float | str] = {
        "task_type": task_type,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "mean_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "throughput_items_per_second": total / elapsed_seconds if elapsed_seconds else 0.0,
    }
    usages = [
        row.get("score_metadata", {}).get("usage")
        for row in predictions
        if isinstance(row.get("score_metadata", {}).get("usage"), dict)
    ]
    if usages:
        input_tokens = sum(int(usage.get("input_tokens", 0) or 0) for usage in usages)
        output_tokens = sum(int(usage.get("output_tokens", 0) or 0) for usage in usages)
        summary.update(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "mean_tokens_per_item": (input_tokens + output_tokens) / total if total else 0.0,
            }
        )
    if predictions and all(row.get("probabilities") is not None for row in predictions):
        summary.update(
            calibration_metrics(
                [row["probabilities"] for row in predictions],
                [int(row["gold_index"]) for row in predictions],
            )
        )
    predicted_indices = [int(row["predicted_index"]) for row in predictions]
    gold_indices = [int(row["gold_index"]) for row in predictions]
    if task_type == "noul":
        if any(index not in {0, 1} for index in [*predicted_indices, *gold_indices]):
            raise ValueError("noul evaluation requires binary indices")
        summary.update(binary_classification_metrics(predicted_indices, gold_indices))
    elif task_type == "score":
        summary.update(ordinal_metrics(predicted_indices, gold_indices))
    elif task_type != "choice":
        raise ValueError(f"unsupported task_type: {task_type}")
    return summary
