if __name__ == "__main__":
    from ranking.train_utils import get_argparser
    ap = get_argparser()
    args = ap.parse_args()

    from ranking.train_utils import init_environment
    init_environment(args)

    # === data ===
    from ranking.datasets import MQDataset, MovielensDataset

    if args.dataset.startswith('mq'):
        if args.dataset == 'mq2007':
            path = '/home/ilya/repos/recsys/data/MQ2007/Fold1'
        elif args.dataset == 'mq2008':
            path = '/home/ilya/repos/recsys/data/MQ2008/Fold1'
        train_dataset = MQDataset(fold_path=path, split='train')
        val_dataset = MQDataset(fold_path=path, split='vali')
    elif args.dataset == 'movielens':
        train_dataset = MovielensDataset(split='train', user_based=args.user_based)
        val_dataset = MovielensDataset(split='test', user_based=args.user_based)
    
    from torch.utils.data import DataLoader, default_collate

    def collate_fn(batch):
        features, targets = default_collate(batch)
        return features.float(), targets        
        
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.n_workers,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate_fn
    )

    # === model and learner ===
    from ranking import RankNet, LambdaRank

    if args.model == 'ranknet':
        model = RankNet(
            input_size=train_dataset.n_features,
            hidden_size=50,
            n_hidden_layers=2,
            temperature=1,
            extra_metrics=args.extra_metrics
        )
    elif args.model == 'lambda-rank':
        model = LambdaRank(
            input_size=train_dataset.n_features,
            hidden_size=50,
            n_hidden_layers=2,
            temperature=1,
            metric_to_optimize=args.metric,
            extra_metrics=args.extra_metrics
        )

    from ranking import Learner, LearnerConfig

    config = LearnerConfig(
        max_lr=args.max_lr,
        lr_div_factor=args.lr_div_factor,
        batch_size=args.batch_size,
        warmup_pct=args.warmup_pct,
        n_epochs=args.n_epochs,
        steps_per_epoch=len(train_dataset),
        lr_decay=args.lr_decay
    )

    learner = Learner(model, config)

    # === trainer ===
    from ranking.train_utils import train

    train(learner, train_loader, val_loader, args)