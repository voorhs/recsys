class LightningCkptLoadable:
    def load_checkpoint(self, path_to_ckpt, learner_class, map_location=None):
        model = learner_class.load_from_checkpoint(
            path_to_ckpt,
            map_location=map_location,
            model=self,
            config=learner_class.get_default_config(),
        ).model
        
        self.load_state_dict(model.state_dict())


class HParamsPuller:
    def get_hparams(self):
        res = {}
        for attr, val in vars(self).items():
            if hasattr(val, 'get_hparams'):
                tmp = val.get_hparams()
                tmp = self.add_prefix(tmp, attr)
                res.update(tmp)
            elif isinstance(val, (int, float, str, bool)):
                res[attr] = val
        return res
    
    @staticmethod
    def add_prefix(dct, prefix):
        res = {}
        for key, val in dct.items():
            res[f'{prefix}.{key}'] = val
        return res


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_argparser():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', dest='model', choices=['ranknet', 'lambda-rank', 'mart', 'lambda-mart'], default='ranknet')
    ap.add_argument('--name', dest='name', default=None)
    ap.add_argument('--seed', dest='seed', default=0, type=int)
    ap.add_argument('--n_workers', dest='n_workers', default=8, type=int)
    ap.add_argument('--n_epochs', dest='n_epochs', default=10, type=int)
    ap.add_argument('--interval', dest='interval', default=1, type=int)
    ap.add_argument('--logdir', dest='logdir', default='../logs')
    ap.add_argument('--logger', dest='logger', choices=['tb', 'wb', 'none'], default='tb')
    ap.add_argument('--batch_size', dest='batch_size', default=4, type=int)
    ap.add_argument('--max_lr', dest='max_lr', default=1e-3, type=float)
    ap.add_argument('--warmup_pct', dest='warmup_pct', default=.1, type=int)
    ap.add_argument('--lr_div_factor', dest='lr_div_factor', default=1e3, type=float)
    ap.add_argument('--resume-training-from', dest='resume_from', default=None)
    ap.add_argument('--load-weights-from', dest='weights_from', default=None)
    return ap


def train(learner, train_loader, val_loader, args):
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

    if args.logger != 'none':
        checkpoint_callback = ModelCheckpoint(
            monitor='val_metric',
            # save_last=True,
            save_top_k=1,
            # every_n_epochs=1,
            mode='max',
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [checkpoint_callback, lr_monitor]
    else:
        callbacks = None

    import lightning.pytorch as pl
    import os
    if args.logger == 'tb':
        Logger = pl.loggers.TensorBoardLogger
        logdir = os.path.join(args.logdir, 'tb')
    elif args.logger == 'wb':
        Logger = pl.loggers.WandbLogger
        logdir = os.path.join(args.logdir, 'wb')
    elif args.logger == 'none':
        Logger = lambda **kwargs: False
        logdir = None
    
    logger = Logger(
        save_dir=logdir,
        name=args.name
    )

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        # max_time={'minutes': 60},
        
        # max_time={'minutes': 10},

        # hardware settings
        accelerator='gpu',
        deterministic=False,
        # precision="16-mixed",

        # logging and checkpointing
        # val_check_interval=args.interval,
        check_val_every_n_epoch=args.interval,
        logger=logger,
        enable_progress_bar=False,
        profiler=None,
        callbacks=callbacks,
        # log_every_n_steps=5,

        # check if model is implemented correctly
        overfit_batches=False,

        # check training_step and validation_step doesn't fail
        fast_dev_run=False,
        num_sanity_val_steps=False
    )

    if args.resume_from is None:
        trainer.validate(learner, val_loader)

    from datetime import datetime
    print('Started at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    trainer.fit(
        learner, train_loader, val_loader,
        ckpt_path=args.resume_from
    )

    print('Finished at', datetime.now().strftime("%H:%M:%S %d-%m-%Y"))

    trainer.validate(learner, val_loader)


def init_environment(args):
    import torch
    torch.set_float32_matmul_precision('medium')

    seed_everything(args.seed)
