from torch.optim import Optimizer
from tqdm import tqdm
from topofm.callbacks import EarlyStopping, ModelCheckpoint
from topofm.data.data_loaders import TrainFMDataLoader, TestFMDataLoader
from topofm.engine.evaluation import evaluate
from topofm.utils import preserve_mode
from topofm.metrics import FMLoss
from topofm.models import Model
from topofm.loggers import Logger
from topofm.odes import ODE
from topofm.frames import Coordinates
from topofm.spaces import Space


def train(
    model: Model,
    ode: ODE, 
    train_loader: TrainFMDataLoader,
    val_loader: TestFMDataLoader,
    optimizer: Optimizer,
    early_stopping: EarlyStopping,
    model_checkpoint: ModelCheckpoint,
    logger: Logger,
    space: Space,
    eval_coords: Coordinates | str,
    val_every_n_samples: int | None = None,
) -> None:
    """
    Train a model. At every validation step, early stopping is checked and 
    the model is checkpointed if it is the best so far.
    
    Args:
        model: The model to train.
        ode: The ODE to use.
        train_loader: The training data loader.
        val_loader: The validation data loader.
        optimizer: The optimizer to use.
        early_stopping: The early stopping callback.
        model_checkpoint: The model checkpoint callback.
        logger: The logger to use.
        space: The space to use.
        eval_coords: The coordinates to use for evaluation.
        val_every_n_samples: The frequency of validation steps.
    Returns:
        None
    """
    eval_coords = Coordinates(eval_coords)
    with preserve_mode(model):
        model.train()

        criterion = FMLoss(ode=ode, model=model)

        if val_every_n_samples is None:
            val_every_n_samples = float('inf')

        samples_processed = 0
        next_val_threshold = val_every_n_samples

        pbar = tqdm(train_loader, desc="Training")        
        for x0, x1 in pbar:
            # Training step
            optimizer.zero_grad()
            loss = criterion(x0, x1)
            loss.backward()
            optimizer.step()
            
            # Log training metrics
            samples_processed += len(x0)
            pbar.set_postfix({"loss": loss.item()})
            logger.log({"train/loss": loss.item()}, step=samples_processed)

            # Maybe validation step
            if samples_processed >= next_val_threshold:
                metrics = evaluate(
                    model=model,
                    ode=ode,
                    data_loader=val_loader,
                    train_space=space,
                    eval_coords=eval_coords,
                )

                # Log validation metrics
                logger.log({f'val/{m}': v for m, v in metrics.items()}, step=samples_processed)

                # Model checkpoint and early stopping callbacks
                model_checkpoint.save_if_best(model, metrics)
                early_stopping.check(metrics)
                
                if early_stopping.should_stop:
                    break

                next_val_threshold += val_every_n_samples