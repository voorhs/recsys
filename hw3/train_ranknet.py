if __name__ == "__main__":
    from neuralcolfil.train_utils import config_to_argparser, retrieve_fields, TrainerConfig
    from neuralcolfil.learners import RankLearner as Learner, RankLearnerConfig as LearnerConfig
    
    ap = config_to_argparser([LearnerConfig, TrainerConfig])
    ap.add_argument('--inference', dest='inference', type=bool, default=False)
    args = ap.parse_args()
    
    learner_config = LearnerConfig(**retrieve_fields(args, LearnerConfig))
    trainer_config = TrainerConfig(**retrieve_fields(args, TrainerConfig))

    from neuralcolfil.train_utils import init_environment
    init_environment(trainer_config.seed)

    # === data ===
    from neuralcolfil.datasets import RekkoRanking
    
    train_dataset = RekkoRanking(
        feed_size=learner_config.feed_size,
        split='train',
    )
    val_dataset = RekkoRanking(
        feed_size=learner_config.feed_size,
        split='val',
    )
    n_users = len(train_dataset.df.user_id.unique())
    n_items = len(train_dataset.df.item_id.unique())

    learner_config.total_steps = len(train_dataset) * trainer_config.n_epochs
    
    from torch.utils.data import DataLoader, default_collate

    def collate_fn(batch):
        batch = default_collate(batch)
        return [x.squeeze(dim=0) for x in batch]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=trainer_config.n_workers,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=trainer_config.n_workers,
        collate_fn=collate_fn
    )

    # === model and learner ===
    from neuralcolfil.models import RankNet

    input_size = train_dataset[0][0].shape[1]
    model = RankNet(
        input_size=input_size,
        hidden_size=64,
        n_hidden_layers=5,
        temperature=1
    )

    learner = Learner(model, learner_config)

    # === trainer ===
    from neuralcolfil.train_utils import train, validate

    if not args.inference:
        train(learner, train_loader, val_loader, trainer_config)
    else:
        validate(learner, val_loader, trainer_config)
