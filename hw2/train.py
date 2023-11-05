if __name__ == "__main__":
    from ranking.train_utils import get_argparser
    ap = get_argparser()
    args = ap.parse_args()

    from ranking.train_utils import init_environment
    init_environment(args)

    # === data ===
    from ranking.datasets import MQDataset
    path = '/home/ilya/repos/recsys/data/MQ2007/Fold1'
    train_dataset = MQDataset(fold_path=path, split='train')
    val_dataset = MQDataset(fold_path=path, split='vali')
    
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
    from ranking import RankNet

    if args.model == 'ranknet':
        model = RankNet(
            input_size=46,
            hidden_size=10,
            temperature=1
        )

    from ranking import Learner, LearnerConfig

    config = LearnerConfig(
        max_lr=args.max_lr,
        lr_div_factor=args.lr_div_factor,
        batch_size=args.batch_size,
        warmup_pct=args.warmup_pct,
        n_epochs=args.n_epochs,
        steps_per_epoch=len(train_dataset)
    )

    learner = Learner(model, config)

    # === trainer ===
    from ranking.train_utils import train

    train(learner, train_loader, val_loader, args)