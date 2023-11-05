from collections import defaultdict
import os
from torch.utils.data import Dataset
import pandas as pd


def load_mq(fold_path, split):
    path_in = os.path.join(fold_path, split + '.txt')
    dataset = defaultdict(list)
    for row in open(path_in, 'r').readlines():
        no_comment, comment = row.split('#')
        features = no_comment.split()
        dataset['relevance_label'].append(int(features[0]))
        for feat in features[1:]:
            feat_name, feat_val = feat.split(':')
            dataset[feat_name].append(float(feat_val))
        comment = comment.split()
        dataset['docid'].append(comment[2])
        dataset['inc'].append(float(comment[5]))
        dataset['prob'].append(float(comment[8]))
    return pd.DataFrame(dataset)


class MQDataset(Dataset):
    def __init__(self, fold_path, split):
        dataset = load_mq(fold_path, split)
        query_ids = dataset['qid'].unique().tolist()
        batches = []
        for qid in query_ids:
            batch = dataset[dataset['qid'] == qid]
            targets = batch['relevance_label'].to_numpy()
            if targets.sum() > 0:
                batches.append(batch)
        self.dataset = pd.concat(batches)
        self.query_ids = self.dataset['qid'].unique().tolist()

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, i):
        qid = self.query_ids[i]
        batch = self.dataset[self.dataset['qid'] == qid]
        targets = batch['relevance_label'].to_numpy()
        features = batch.drop(columns=['relevance_label', 'qid', 'docid', 'inc', 'prob']).to_numpy()
        return features, targets
