import torch
from torch import Tensor
from topofm.config import DATA_DIR
from topofm.data.datasets.fm_dataset import FMDataset
from topofm.distributions.boundary.empirical_distribution import EmpiricalDistribution
from topofm.spaces import Space
from topofm.frames.frame import Coordinates
from topofm.odes import ODE


SINGLE_CELL_DATA_DIR = DATA_DIR / "single_cell"
    

def load_single_cell_data(data_dir: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = np.load(os.path.join(data_dir, 'mu0.npy'))
    x1 = np.load(os.path.join(data_dir, 'mu4.npy'))
    return torch.as_tensor(x0), torch.as_tensor(x1)


def load_single_cell_eigenpairs(data_dir: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvectors = np.load(os.path.join(data_dir, 'L_eigenvectors.npy'))
    eigenvalues = np.load(os.path.join(data_dir, 'L_eigenvalues.npy'))
    return torch.as_tensor(eigenvectors, device=torch.get_default_device(), dtype=torch.get_default_dtype()), torch.as_tensor(eigenvalues, device=torch.get_default_device(), dtype=torch.get_default_dtype())


def load_single_cell_true_times(data_dir: str | None = None) -> torch.Tensor:
    return torch.as_tensor(np.load(os.path.join(data_dir, 'label.npy')))


def load_single_cell_phate(data_dir: str | None = None) -> torch.Tensor:
    return torch.as_tensor(np.load(os.path.join(data_dir, 'coord.npy')))



class SingleCellGraph(Space):
    # TODO Inherit from spaces.Graph, which should store a graph structure
    # optionally with nodes, edges, and node coordinates.
    def __init__(
        self, 
        coords: Coordinates | str, 
        dtype: torch.dtype, 
        device: torch.device,
    ) -> None:
        eigvals = SingleCellGraph.load_eigvals().to(device=device, dtype=dtype)
        eigvecs = SingleCellGraph.load_eigvecs().to(device=device, dtype=dtype)
        super().__init__(eigvals=eigvals, eigvecs=eigvecs, coords=coords)

    @staticmethod
    def load_eigvals() -> Tensor:
        return torch.load(SINGLE_CELL_DATA_DIR / 'eigenvalues.pt', map_location="cpu")

    @staticmethod
    def load_eigvecs() -> Tensor:
        return torch.load(SINGLE_CELL_DATA_DIR / 'eigenvectors.pt', map_location="cpu")


class SingleCellDataset(FMDataset):

    @staticmethod
    def load_data() -> tuple[Tensor, Tensor]:
        x0 = torch.load(SINGLE_CELL_DATA_DIR / 'x0.pt', map_location="cpu")
        x1 = torch.load(SINGLE_CELL_DATA_DIR / 'x1.pt', map_location="cpu")
        return x0, x1

    @classmethod
    def from_disk(
        cls,
        space: SingleCellGraph,
        ode: ODE,
        device: torch.device, 
        dtype: torch.dtype,
    ) -> "SingleCellDataset":
        x0, x1 = SingleCellDataset.load_data()
        x0 = x0.to(device=device, dtype=dtype)
        x1 = x1.to(device=device, dtype=dtype)
        mu0 = EmpiricalDistribution.from_standard(x0, space=space)
        mu1 = EmpiricalDistribution.from_standard(x1, space=space)
        return cls(
            mu0=mu0,
            mu1=mu1,
        )
    
    def _split(self, split: tuple[float, float, float] = (0.7, 0.1, 0.2)) -> tuple["SingleCellDataset", "SingleCellDataset", "SingleCellDataset"]:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            split (tuple[float, float, float]): The ratio of the dataset to be used for training, validation, and test.

        Returns:
            tuple[SingleCellDataset, SingleCellDataset, SingleCellDataset]: The training, validation, and test datasets.
        """
        return self, self, self
