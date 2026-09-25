from typing import Dict, List, Set, Union

def compute_entity_f_beta(
    true_ids: Set[str], 
    pred_ids: Set[str], 
    beta: float = 0.5
) -> float:
    """
    Computes F_beta score for a single Source 1 entity.
    Includes singleton logic:
    - If true_ids is empty:
        - 1.0 if pred_ids is empty (correct singleton prediction)
        - 0.0 if pred_ids is not empty (false merge / false positive)
    - If true_ids is non-empty:
        - 0.0 if pred_ids is empty (missed match)
        - F_beta formula otherwise.
    """
    beta_sq = beta ** 2
    
    # Singleton cases (no true matches)
    if len(true_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0
    
    # Non-singleton case but predicted empty
    if len(pred_ids) == 0:
        return 0.0
    
    tp = len(true_ids.intersection(pred_ids))
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)
    
    if tp == 0:
        return 0.0
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    numerator = (1 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator


def compute_macro_f05(
    ground_truth_dict: Dict[str, Union[List[str], Set[str]]],
    predictions_dict: Dict[str, Union[List[str], Set[str]]]
) -> float:
    """
    Computes the competition Macro-Averaged F_0.5 score across all Source 1 entities.
    Pure standard library implementation.
    """
    if not ground_truth_dict:
        return 0.0
        
    scores = []
    for s1_id, true_matches in ground_truth_dict.items():
        true_set = set(true_matches) if isinstance(true_matches, (list, set)) else set()
        pred_matches = predictions_dict.get(s1_id, [])
        pred_set = set(pred_matches) if isinstance(pred_matches, (list, set)) else set()
        
        score = compute_entity_f_beta(true_set, pred_set, beta=0.5)
        scores.append(score)
        
    return sum(scores) / len(scores)


if __name__ == "__main__":
    # Test with the competition prompt example:
    # Pred: [S2-00047, S2-00193, S3-00812], Truth: [S2-00047, S3-00812]
    # Precision = 2/3, Recall = 1.0 -> F_0.5 = 0.714
    example_pred = {"S2-00047", "S2-00193", "S3-00812"}
    example_true = {"S2-00047", "S3-00812"}
    score = compute_entity_f_beta(example_true, example_pred, beta=0.5)
    print(f"Test Example Score: {score:.3f} (Expected: ~0.714)")
    assert abs(score - 0.7142857) < 1e-4, "Prompt example calculation mismatch"
    
    # Singleton test
    assert compute_entity_f_beta(set(), set(), beta=0.5) == 1.0, "Singleton empty should score 1.0"
    assert compute_entity_f_beta(set(), {"S2-99"}, beta=0.5) == 0.0, "Singleton false positive should score 0.0"
    
    # Macro average test
    gt_map = {
        "S1-1": ["S2-1", "S3-1"],
        "S1-2": [] # singleton
    }
    pred_map = {
        "S1-1": ["S2-1", "S3-1"], # 1.0
        "S1-2": [] # 1.0
    }
    macro_score = compute_macro_f05(gt_map, pred_map)
    print(f"Macro F0.5 Test: {macro_score:.4f} (Expected: 1.0000)")
    assert macro_score == 1.0
    print("All metric tests passed with zero external dependencies!")
