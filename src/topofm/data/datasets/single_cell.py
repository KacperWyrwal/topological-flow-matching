
SINGLE_CELL_URL = "https://data.mendeley.com/public-files/datasets/hhny5ff7yj/files/d82698f4-d143-442f-9a41-10be8ad02584/file_downloaded"


def download_single_cell_data(data_dir: str):
    """
    The single-cell dataset is the ebdata_v3.h5ad file.
    """
    os.makedirs(data_dir, exist_ok=True)
    response = requests.get(SINGLE_CELL_URL, timeout=60)
    response.raise_for_status()
    with open(os.path.join(data_dir, 'ebdata_v3.h5ad'), 'wb') as f:
        f.write(response.content)
    print(f"Downloaded single-cell data to {data_dir}")
    

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