if __name__ == "__main__":
    from src.train_utils import config_to_argparser, retrieve_fields, TrainerConfig
    from src.learners import ColFilLearnerConfig as LearnerConfig
    
    ap = config_to_argparser([LearnerConfig, TrainerConfig])
    ap.add_argument('--model', dest='model', choices=['gmf', 'mlp', 'ncf', 'ncfr', 'transformer', 'hybrid'], default='hybrid')
    ap.add_argument('--eval', dest='eval', action='store_true')
    ap.add_argument('--embedding-dim', dest='embedding_dim', type=int, default=8)
    ap.add_argument('--gmf-weights', dest='gmf_weights', default='/home/ilya/repos/recsys/hw4/logs/tb/gmf-8/version_0/checkpoints/epoch=9-step=478550.ckpt')
    ap.add_argument('--mlp-weights', dest='mlp_weights', default='/home/ilya/repos/recsys/hw4/logs/tb/mlp-8-v2/version_0/checkpoints/epoch=9-step=7470.ckpt')
    ap.add_argument('--with-hidden', dest='with_hidden', action='store_true')
    args = ap.parse_args()
    
    learner_config = LearnerConfig(**retrieve_fields(args, LearnerConfig))
    trainer_config = TrainerConfig(**retrieve_fields(args, TrainerConfig))

    from src.train_utils import init_environment
    init_environment(trainer_config.seed)

    # === data ===
    if args.model == 'hybrid':
        from src.datasets import RekkoHybrid as Dataset
    else:
        from src.datasets import RekkoImplicit as Dataset
    
    train_dataset = Dataset(
        n_negatives=learner_config.n_negatives_train,
        split='train',
    )
    val_dataset = Dataset(
        n_negatives=learner_config.n_negatives_val,
        split='val',
    )
    n_users = len(train_dataset.df.user_id.unique())
    n_items = len(train_dataset.df.item_id.unique())

    learner_config.total_steps = len(train_dataset) * trainer_config.n_epochs // learner_config.batch_size
    
    from torch.utils.data import DataLoader
    import torch

    def collate_fn(batch):
        return torch.LongTensor(batch)
    
    def hybrid_collate_fn(batch):
        pre_results = [list() for _ in range(len(batch[0]))]
        for arrays in batch:
            for i, arr in enumerate(arrays):
                pre_results[i].append(torch.from_numpy(arr))
        return [torch.cat(arrs, dim=0) for arrs in pre_results]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=learner_config.batch_size,
        shuffle=True,
        num_workers=trainer_config.n_workers,
        drop_last=True,
        collate_fn=collate_fn if args.model != 'hybrid' else hybrid_collate_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=learner_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.n_workers,
        collate_fn=collate_fn if args.model != 'hybrid' else hybrid_collate_fn
    )

    # === model and learner ===
    from src.models import *

    def load_gmf(pretrained):
        gmf = GMF(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=args.embedding_dim
        )
        if pretrained:
            gmf.load_checkpoint(path_to_ckpt=args.gmf_weights)
        return gmf

    def load_mlp(pretrained):
        e_dim = args.embedding_dim
        mlp = MLP(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=e_dim,
            hidden_sizes=[e_dim, e_dim, e_dim // 2, e_dim // 2]
        )
        if pretrained:
            mlp.load_checkpoint(path_to_ckpt=args.mlp_weights)
        return mlp

    if args.model == 'gmf':
        model = load_gmf(pretrained=False)
    elif args.model == 'mlp':
        model = load_mlp(pretrained=False)
    elif args.model == 'ncf':
        gmf = load_gmf(pretrained=True)
        mlp = load_mlp(pretrained=True)
        model = NCF(gmf, mlp)
    elif args.model == 'transformer':
        gmf = load_gmf(pretrained=True)
        mlp = load_mlp(pretrained=True)
        model = TransformerRecommender(
            config=None,
            embed_user_gmf=gmf.embed_user,
            embed_item_gmf=gmf.embed_item,
            embed_user_mlp=mlp.embed_user,
            embed_item_mlp=mlp.embed_item
        )
    elif args.model == 'hybrid':
        gmf = load_gmf(pretrained=True)
        mlp = load_mlp(pretrained=True)
        model = HybridRecommender(gmf, mlp, with_hidden_layers=args.with_hidden)
    else:
        raise ValueError(f'unknown model {args.model}')


    if args.model == 'hybrid':
        from src.learners import HybridLearner as Learner
    else:
        from src.learners import ColFilLearner as Learner
    
    learner = Learner(model, learner_config)

    # === trainer ===
    from src.train_utils import train, validate

    if not args.eval:
        train(learner, train_loader, val_loader, trainer_config)
    else:
        validate(learner, val_loader, trainer_config)
