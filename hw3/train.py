if __name__ == "__main__":
    from neuralcolfil.train_utils import config_to_argparser, retrieve_fields, TrainerConfig
    from neuralcolfil.learner import LearnerConfig
    
    ap = config_to_argparser([LearnerConfig, TrainerConfig])
    ap.add_argument('--model', dest='model', choices=['gmf', 'mlp', 'ncf'], default='gmf')
    ap.add_argument('--embedding-dim', dest='embedding_dim', type=int, default=16)
    args = ap.parse_args()
    
    learner_config = LearnerConfig(**retrieve_fields(args, LearnerConfig))
    trainer_config = TrainerConfig(**retrieve_fields(args, TrainerConfig))

    from neuralcolfil.train_utils import init_environment
    init_environment(args.seed)

    # === data ===
    from neuralcolfil.dataset import RekkoDataset
    
    train_dataset = RekkoDataset(
        n_negatives=learner_config.n_negatives_train,
        split='train',
    )
    val_dataset = RekkoDataset(
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
        num_workers=args.n_workers,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate_fn
    )

    # === model and learner ===
    from neuralcolfil.models import MLPCollaborativeFilterer, GeneralizedMatrixFactorization, NeuralCollaborativeFilterer

    if args.model == 'gmf':
        model = GeneralizedMatrixFactorization(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=args.embedding_dim
        )
    elif args.model == 'mlp':
        model = MLPCollaborativeFilterer(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=args.embedding_dim,
            hidden_sizes=[32, 32, 32]
        )
    elif args.model == 'ncf':
        gmf = GeneralizedMatrixFactorization(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=args.embedding_dim
        )
        gmf.load_checkpoint(
            path_to_ckpt='/home/ilya/repos/recsys/hw3/logs/tb/gmf/checkpoints/last.ckpt'
        )
        mlp = MLPCollaborativeFilterer(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=args.embedding_dim,
            hidden_sizes=[32, 32, 32]
        )
        mlp.load_checkpoint(
            path_to_ckpt='/home/ilya/repos/recsys/hw3/logs/tb/mlp/checkpoints/epoch=9-step=478550.ckpt'
        )
        model = NeuralCollaborativeFilterer(gmf, mlp)
    else:
        raise ValueError(f'unknown model {args.model}')

    from neuralcolfil.learner import Learner, LearnerConfig

    learner = Learner(model, learner_config)

    # === trainer ===
    from neuralcolfil.train_utils import train

    train(learner, train_loader, val_loader, trainer_config)
