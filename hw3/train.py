if __name__ == "__main__":
    from neuralcolfil.train_utils import config_to_argparser, retrieve_fields, TrainerConfig
    from neuralcolfil.learners import ColFilLearner as Learner, ColFilLearnerConfig as LearnerConfig
    
    ap = config_to_argparser([LearnerConfig, TrainerConfig])
    ap.add_argument('--model', dest='model', choices=['gmf', 'mlp', 'ncf', 'ncfr'], default='gmf')
    ap.add_argument('--embedding-dim', dest='embedding_dim', type=int, default=16)
    args = ap.parse_args()
    
    learner_config = LearnerConfig(**retrieve_fields(args, LearnerConfig))
    trainer_config = TrainerConfig(**retrieve_fields(args, TrainerConfig))

    from neuralcolfil.train_utils import init_environment
    init_environment(trainer_config.seed)

    # === data ===
    from neuralcolfil.datasets import RekkoImplicit
    
    train_dataset = RekkoImplicit(
        n_negatives=learner_config.n_negatives_train,
        split='train',
    )
    val_dataset = RekkoImplicit(
        n_negatives=learner_config.n_negatives_val,
        split='val',
    )
    n_users = len(train_dataset.df.user_id.unique())
    n_items = len(train_dataset.df.item_id.unique())

    learner_config.total_steps = len(train_dataset) * trainer_config.n_epochs
    
    from torch.utils.data import DataLoader
    import torch

    def collate_fn(batch):
        return torch.LongTensor(batch)    

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=learner_config.batch_size,
        shuffle=True,
        num_workers=trainer_config.n_workers,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.n_workers,
        collate_fn=collate_fn
    )

    # === model and learner ===
    from neuralcolfil.models import MLP, GMF, NCF, RankNet, NCFR

    if args.model == 'gmf':
        model = GMF(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=args.embedding_dim
        )
    elif args.model == 'mlp':
        model = MLP(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=args.embedding_dim,
            hidden_sizes=[32, 32, 32]
        )
    elif args.model == 'ncf':
        gmf = GMF(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=args.embedding_dim
        )
        gmf.load_checkpoint(
            path_to_ckpt='/home/ilya/repos/recsys/hw3/logs/tb/gmf/version_4/checkpoints/epoch=9-step=478550.ckpt'
        )
        mlp = MLP(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=args.embedding_dim,
            hidden_sizes=[32, 32, 32]
        )
        mlp.load_checkpoint(
            path_to_ckpt='/home/ilya/repos/recsys/hw3/logs/tb/mlp/version_2/checkpoints/epoch=9-step=478550.ckpt'
        )
        model = NCF(gmf, mlp)
    elif args.model == 'ncfr':
        gmf = GMF(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=args.embedding_dim
        )
        gmf.load_checkpoint(
            path_to_ckpt='/home/ilya/repos/recsys/hw3/logs/tb/gmf/version_4/checkpoints/epoch=9-step=478550.ckpt'
        )
        mlp = MLP(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=args.embedding_dim,
            hidden_sizes=[32, 32, 32]
        )
        mlp.load_checkpoint(
            path_to_ckpt='/home/ilya/repos/recsys/hw3/logs/tb/mlp/version_2/checkpoints/epoch=9-step=478550.ckpt'
        )
        ranknet = RankNet(
            input_size=args.embedding_dim,
            hidden_size=32,
            n_hidden_layers=3,
            temperature=1
        )
        ranknet.load_checkpoint(
            path_to_ckpt='/home/ilya/repos/recsys/hw3/logs/tb/test/version_5/checkpoints/epoch=0-step=2484.ckpt'
        )
        model = NCFR(gmf, mlp, ranknet)
    else:
        raise ValueError(f'unknown model {args.model}')

    learner = Learner(model, learner_config)

    # === trainer ===
    from neuralcolfil.train_utils import train

    train(learner, train_loader, val_loader, trainer_config)
