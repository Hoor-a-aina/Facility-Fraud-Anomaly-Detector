import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class TorchAutoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, 6), nn.ReLU(), nn.Linear(6, 3), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, n_features))
    def forward(self, x):
        return self.decoder(self.encoder(x))

def fit_score_models(X_train, X_val, X_test, y_train=None, seed=42):
    """Fit detectors on training data only; higher scores always mean more anomalous."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    imputer = SimpleImputer(strategy="median")
    Xtr = imputer.fit_transform(X_train)
    Xva = imputer.transform(X_val)
    Xte = imputer.transform(X_test)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    iso = IsolationForest(n_estimators=250, contamination="auto", random_state=seed, n_jobs=-1)
    iso.fit(Xtr_s)
    scores = {
        "Isolation Forest": (
            -iso.score_samples(Xva_s),
            -iso.score_samples(Xte_s),
        )
    }

    svm = OneClassSVM(kernel="rbf", nu=0.015, gamma="scale")
    svm.fit(Xtr_s)
    scores["One-Class SVM"] = (
        -svm.decision_function(Xva_s).ravel(),
        -svm.decision_function(Xte_s).ravel(),
    )

    # Autoencoder learns only normal examples from the training period.
    normal_mask = np.asarray(y_train == 0) if y_train is not None else np.ones(len(Xtr_s), dtype=bool)
    X_normal = Xtr_s[normal_mask]
    if len(X_normal) < 10:
        raise ValueError("Not enough normal training rows to fit autoencoder")
    device = torch.device("cpu")
    model = TorchAutoencoder(Xtr_s.shape[1]).to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X_normal, dtype=torch.float32)),
                        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(35):
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
    model.eval()
    def reconstruction_error(arr):
        with torch.no_grad():
            x = torch.tensor(arr, dtype=torch.float32, device=device)
            return torch.mean((model(x) - x) ** 2, dim=1).cpu().numpy()
    scores["PyTorch Autoencoder"] = (
        reconstruction_error(Xva_s),
        reconstruction_error(Xte_s),
    )
    return scores
