import os
from dataclasses import dataclass
from typing import Optional

from everything import StandardFrame
from sklearn.model_selection import KFold
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from topofm import (
    HeatBMTSDE,
    SpectralFrame,
    OTBatchSampler,
    make_ot_solver,
    make_sde_solver,
    make_time_sampler,
    make_optimizer,
    make_ema,
    make_objective,
    fit,
    GCN,
    ResidualNN,
    load_brain_data, 
    load_brain_laplacian, 
    MatchingTensorDataset,
    MatchingDataLoader,
)


def _build_model(cfg: DictConfig, data_dim: int, laplacian: Optional[torch.Tensor] = None) -> torch.nn.Module:
    if cfg.model.name == "residual_nn":
        return ResidualNN(data_dim=data_dim, hidden_dim=cfg.model.hidden_dim, time_embed_dim=cfg.model.time_embed_dim, num_res_block=cfg.model.num_res_block)
    elif cfg.model.name == "gcn":
        assert laplacian is not None, "laplacian is required for GCN"
        return GCN(laplacian=laplacian, hidden_dim=cfg.model.hidden_dim, time_embed_dim=cfg.model.time_embed_dim)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")


def _build_sde(cfg: DictConfig, eigenvalues: torch.Tensor) -> HeatBMTSDE:
    return HeatBMTSDE(eigenvalues=eigenvalues, c=cfg.sde.c, sigma=torch.tensor(cfg.sde.sigma))


def _build_laplacian(cfg: DictConfig) -> torch.Tensor:
    if cfg.data.name == 'brain':
        return load_brain_laplacian()
    raise ValueError(f"Unsupported tensor dataset name: {cfg.data.name}")


def _build_frame(cfg: DictConfig) -> torch.Tensor:
    if cfg.sde.topological is False:
        return StandardFrame()
    else:
        L = _build_laplacian(cfg)
        return SpectralFrame(L)


def _build_dataset(cfg: DictConfig, frame: SpectralFrame | None = None):
    frame = _build_frame(cfg)
    if cfg.data.name == "brain":
        x0, x1 = load_brain_data() # TODO pass option to retrieve the training split
        return MatchingTensorDataset(x0, x1, frame=frame)
    raise ValueError(f"Unknown dataset: {cfg.data.name}")


def run_full(

) -> torch.nn.Module:
    pass


def run_validation(
    cfg: DictConfig,
    *, 
    dataset: MatchingTensorDataset, 

) -> torch.nn.Module:
    assert cfg.data.type == 'tensor', "Only tensor datasets are foldable."
    metric_name = cfg.trainer.validation_metric
    model = _build_model(cfg, ...)
    early_stopping = ... 
    wnb_logger = ...
  
    validation_
    for fold, ds in k_fold_split(dataset):
        ds_train, ds_eval = train_test_split(ds)
        history = fit(...)
        # evaluate the model 


    cv_metric = float(sum(fold_metrics) / len(fold_metrics))
    return {"cv_metric": cv_metric, "fold_metrics": fold_metrics, "histories": histories}


@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    assert cfg.data.type == 'tensor', "Currently only tensor datasets are supported."

    # Prelimiaries 
    torch.manual_seed(cfg.run.seed)
    torch.set_default_device(torch.device(cfg.run.device))
    torch.set_default_dtype(torch.dtype(cfg.run.dtype))

    # Common terms of full and cross-validation runs 
    frame = _build_frame(cfg)
    dataset = _build_dataset(cfg, frame=frame)
    sde = _build_sde(cfg, frame.eigenvalues)

    if cfg.train.validation is False: # TODO Maybe instead have cfg.train.mode == 'validation' and then a separate config for that mode 
        run_validation(cfg, dataset=dataset, sde=sde)
    else:
        run_full(cfg, ...)

    # TODO 


if __name__ == "__main__":
    main()


