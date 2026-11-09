"""Step 7: Q04 (P§5) - matching removes a contrast-only task difference; a real drop survives."""
from start_vision.questions import q04
from tests.synthetic.questions import ds006171_table


def test_questions_q04_matching_removes_contrast_confound() -> None:
    per = q04.subject_task_auc(ds006171_table(n_subjects=40, n_trials=800,
                                              contrast_auc_only=True, seed=8))
    w = per.pivot_table(index="subject", columns="task", values="auc_matched")
    assert abs(w.mean().mean() - 0.5) < 0.04
    assert per["auc_unmatched"].mean() > 0.55
    r = q04.group(per)
    assert r["primary_tost"]["p"] < 0.05


def test_questions_q04_true_drop_survives_matching() -> None:
    per = q04.subject_task_auc(ds006171_table(task_drop=0.8, seed=9))
    r = q04.group(per)
    assert r["informative_diff"]["mean"] < -0.05 and r["primary_tost"]["p"] > 0.05
