# Importing libraries
import os
import yaml
import torch
from munch import Munch
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, CometLogger, TensorBoardLogger, WandbLogger
from PW_FT_classification.src import algorithms
from PW_FT_classification.src import datasets


def main(
        config:str='./classify/config_classify.yaml',
        project:str='SINClassifierV2',
        gpus:str='0', 
        logger_type:str='csv',
        evaluate:str=None,
        np_threads:str='32',
        session:int=None,
        seed:int=0,
        dev:bool=False,
        mode:str='train_test', #"train_only", "train_test", "val", "test", or "predict"
        predict_root:str=""
    ):
    """
    Main function for training or evaluating a ResNet-50 model using PyTorch Lightning.
    It loads configurations, initializes the model, logger, and other components based on provided arguments.

    Args:
        config (str): Path to the configuration file.
        project (str): Name of the project for logging.
        gpus (str): Comma-separated GPU ids for training.
        logger_type (str): Type of logger to use (wandb, comet, tensorboard, csv).
        evaluate (str): Path to the model checkpoint for evaluation.
        np_threads (str): Number of numpy threads to use.
        session (int): Session number for logging purposes.
        seed (int): Random seed for reproducibility.
        dev (bool): Development mode flag.
        val (bool): Validation mode flag.
        predict (bool): Prediction mode flag.
        predict_root (str): Root directory for prediction outputs.
    """

    # GPU configuration: set up GPUs based on availability and user specification
    gpus = gpus if torch.cuda.is_available() else None
    gpus = [int(i) for i in gpus.split(',')]

    # Environment variable setup for numpy multi-threading
    os.environ["OMP_NUM_THREADS"] = str(np_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(np_threads)
    os.environ["MKL_NUM_THREADS"] = str(np_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(np_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(np_threads)

    # Load and set configurations from the YAML file
    with open(config) as f:
        conf = Munch(yaml.load(f, Loader=yaml.FullLoader))
    conf.evaluate = evaluate
    conf.mode = mode
    conf.predict_root = predict_root
    conf.start_time = datetime.now()
    results_folder = 'results_dev' if dev else 'results'
    conf.save_dir  = './{}/{}/{}_{}'.format(results_folder, project, conf.algorithm, conf.conf_id)
    
    # Set a global seed for reproducibility
    pl.seed_everything(seed)

    # Logger setup based on the specified logger type
    logger = None
    if logger_type == 'csv':
        logger = CSVLogger(
            save_dir=conf.save_dir,
            prefix=project,
            name=None, 
            version=session
        )
    elif logger_type == 'tensorboard':
        logger = TensorBoardLogger(
            save_dir=conf.save_dir,
            prefix=project,
            name=None,
            version=session
        )
    elif logger_type == 'comet':
        logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            save_dir=conf.save_dir,
            project_name=project, 
            experiment_name=None,
        )
    elif logger_type == 'wandb':
        logger = WandbLogger(
            save_dir=conf.save_dir,
            project=project,  
            name=None,
        )

    # Update save_dir based on logger.version
    conf.save_dir = os.path.join(conf.save_dir, f"version_{logger.version}") 

    # Dataset and algorithm loading based on the configuration
    dataset = datasets.__dict__[conf.dataset_name](conf=conf)
    learner = algorithms.__dict__[conf.algorithm](
        conf=conf,
        train_class_counts=dataset.train_class_counts, 
        id_to_labels=dataset.id_to_labels, 
        head_lists=conf.head_lists,
        epoch_thresh=conf.epoch_thresh
    )

    # Callbacks for model checkpointing and learning rate monitoring
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_mac_acc', 
        mode='max', 
        dirpath=conf.save_dir, 
        save_top_k=5, 
        filename='{}'.format(conf.conf_id) + '-{epoch:02d}-{valid_mac_acc:.2f}', 
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer configuration in PyTorch Lightning
    trainer = pl.Trainer(
        max_epochs=conf.num_epochs,
        check_val_every_n_epoch=1, 
        log_every_n_steps=conf.log_interval, 
        accelerator='gpu',
        devices=gpus,
        logger=None if evaluate is not None else logger,
        callbacks=[lr_monitor, checkpoint_callback],
        num_sanity_val_steps=0,
        profiler=None
    )
    # Training, validation, or evaluation execution based on the mode
    if evaluate is not None:
        if conf.mode == 'val':
            trainer.validate(learner, dataloaders=[dataset.val_dataloader()], ckpt_path=evaluate)
        elif conf.mode == 'predict':
            trainer.predict(learner, dataloaders=[dataset.predict_dataloader()], ckpt_path=evaluate)
        elif conf.mode == 'test':
            trainer.test(learner, dataloaders=[dataset.test_dataloader()], ckpt_path=evaluate)
        else:
            print('Invalid mode for evaluation.')
    else:
        if conf.mode == 'train_only':
            trainer.fit(learner, datamodule=dataset, ckpt_path=conf.resume_from_ckpt)
        elif conf.mode == 'train_test': 
            trainer.fit(learner, datamodule=dataset, ckpt_path=conf.resume_from_ckpt)
            trainer.test(learner, dataloaders=[dataset.test_dataloader()], ckpt_path="best")
        else:
            print('Invalid mode for training.')


if __name__ == '__main__':
    main()
