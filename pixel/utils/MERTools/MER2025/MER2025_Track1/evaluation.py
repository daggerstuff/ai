


from sklearn.metrics import f1_score
from toolkit.globals import *
from toolkit.utils.functions import *


def score_calculation(label_csv, submission_csv):

    # 1. read gt
    name2gt = {}
    names = func_read_key_from_csv(label_csv, "name")
    emos = func_read_key_from_csv(label_csv, "discrete")
    for name, emo in zip(names, emos):
        name2gt[name] = emo

    # 2. read pred
    name2pred = {}
    names = func_read_key_from_csv(submission_csv, "name")
    preds = func_read_key_from_csv(submission_csv, "discrete")
    for name, pred in zip(names, preds):
        name2pred[name] = pred
    assert len(name2pred) == 20000

    # 3. score calculation
    process_names = list(name2gt.keys())
    emo_labels = [emo2idx_mer[name2gt[name]] for name in process_names]
    emo_preds = [emo2idx_mer[name2pred[name]] for name in process_names]
    emo_fscore = f1_score(emo_labels, emo_preds, average="weighted")
    print(emo_fscore)
    return emo_fscore


if __name__ == "__main__":
    import fire

    fire.Fire()
