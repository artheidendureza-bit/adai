"""
Dataset utilities for adAI

IMPORTANT: All built-in datasets are SYNTHETIC APPROXIMATIONS for educational purposes.
They mimic the structure and statistics of real datasets but are not the actual data.
For production use or research, please download the original datasets from their official sources.

Available datasets:
- Classification: iris, wine, breast_cancer, adult, heart_disease, bank_marketing
- Regression: boston_housing, diabetes, california_housing, energy
- Image: mnist, fashion_mnist, cifar10, cifar100, imagenet
- Text: imdb, penn_treebank
- Spiking: ssc, shd
- Synthetic: synthetic, synthetic_reg
"""

import numpy as np
from typing import Tuple, Dict, Optional, Union


def _split_and_normalize(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Internal function to split data into train/test and normalize using min-max scaling.
    
    Args:
        X: Features array (n_samples, n_features)
        y: Target array (n_samples,)
        test_size: Fraction of data for test set (default: 0.2)
        random_state: Random seed for reproducibility (optional)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler_params)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    split = int((1 - test_size) * n_samples)
    indices = np.random.permutation(n_samples)
    
    train_idx, test_idx = indices[:split], indices[split:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Min-max normalize X using training set statistics
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_train_norm = (X_train - X_min) / (X_max - X_min + 1e-8)
    X_test_norm = (X_test - X_min) / (X_max - X_min + 1e-8)
    
    # Scale y by training max
    y_max = y_train.max()
    y_train_scaled = y_train / y_max
    y_test_scaled = y_test / y_max
    
    scaler_params = {
        'X_min': X_min,
        'X_max': X_max,
        'y_max': y_max
    }
    
    return X_train_norm, X_test_norm, y_train_scaled, y_test_scaled, scaler_params


def load_data(
    dataset: Optional[str] = None,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Load a dataset or split/normalize provided data.
    
    Args:
        dataset: Name of dataset to load. Options:
            Classification:
            - 'iris': Iris flower classification (150 samples, 4 features, 3 classes)
            - 'wine': Wine classification (178 samples, 13 features, 3 classes)
            - 'breast_cancer': Breast cancer classification (569 samples, 30 features, 2 classes)
            - 'adult' or 'census_income': Adult/Census Income (48842 samples, 14 features, 2 classes)
            - 'heart_disease': Heart Disease (303 samples, 13 features, 2 classes)
            - 'bank_marketing': Bank Marketing (45211 samples, 16 features, 2 classes)
            
            Regression:
            - 'boston_housing': Boston housing regression (506 samples, 13 features)
            - 'diabetes': Diabetes regression (442 samples, 10 features)
            - 'california_housing': California housing regression (20640 samples, 8 features)
            - 'energy': Energy efficiency regression (768 samples, 8 features)
            
            Image (synthetic approximations):
            - 'mnist': MNIST digits (70000 samples, 784 features, 10 classes)
            - 'fashion_mnist': Fashion MNIST (70000 samples, 784 features, 10 classes)
            - 'cifar10' or 'cifar_10': CIFAR-10 (60000 samples, 3072 features, 10 classes)
            - 'cifar100' or 'cifar_100': CIFAR-100 (60000 samples, 3072 features, 100 classes)
            - 'imagenet': ImageNet (10000 samples, 150528 features, 1000 classes)
            
            Text (synthetic approximations):
            - 'imdb' or 'imdb_reviews': IMDB Reviews (50000 samples, 5000 features, 2 classes)
            - 'penn_treebank' or 'ptb': Penn Treebank (42068 samples, 10000 features)
            
            Spiking (synthetic approximations):
            - 'ssc' or 'spiking_speech_commands': Spiking Speech Commands (105829 samples, 700 features, 35 classes)
            - 'shd' or 'spiking_heidelberg_digits': Spiking Heidelberg Digits (20282 samples, 256 features, 20 classes)
            
            Synthetic:
            - 'synthetic' or 'synthetic_classification': Generate synthetic classification data
            - 'synthetic_reg' or 'synthetic_regression': Generate synthetic regression data
            
        X: Features array (if not using dataset). Can be 2D (n_samples, n_features) or 4D (n_samples, h, w, c) for CNN data
        y: Target array (if not using dataset)
        test_size: Fraction of data for test set (default: 0.2)
        random_state: Random seed for reproducibility (optional)
        **kwargs: Additional parameters for specific datasets (e.g., n_samples for synthetic)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler_params)
        
    Examples:
        >>> # Load built-in dataset
        >>> X_train, X_test, y_train, y_test, scaler = load_data('iris')
        
        >>> # Use custom data
        >>> X_train, X_test, y_train, y_test, scaler = load_data(X=X, y=y)
        
        >>> # Generate synthetic data
        >>> X_train, X_test, y_train, y_test, scaler = load_data('synthetic', n_samples=100)
    """
    if dataset is not None:
        X, y = _load_dataset(dataset, random_state=random_state, **kwargs)
    elif X is None or y is None:
        raise ValueError("Either 'dataset' or both 'X' and 'y' must be provided")
    
    # Handle 4D CNN data by flattening, normalizing, then reshaping
    original_shape = X.shape
    if len(X.shape) == 4:
        X = X.reshape(X.shape[0], -1)
    
    X_train, X_test, y_train, y_test, scaler_params = _split_and_normalize(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Reshape back to 4D if original was 4D
    if len(original_shape) == 4:
        X_train = X_train.reshape(-1, *original_shape[1:])
        X_test = X_test.reshape(-1, *original_shape[1:])
    
    return X_train, X_test, y_train, y_test, scaler_params


def _load_dataset(dataset: str, random_state: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Load a specific dataset."""
    if random_state is not None:
        np.random.seed(random_state)
    
    dataset = dataset.lower().replace('-', '_').replace(' ', '_')
    
    # Classification datasets (synthetic approximations)
    if dataset == 'iris':
        return _load_iris()
    elif dataset == 'wine':
        return _load_wine()
    elif dataset == 'breast_cancer':
        return _load_breast_cancer()
    elif dataset == 'adult' or dataset == 'census_income':
        return _load_adult()
    elif dataset == 'heart_disease':
        return _load_heart_disease()
    elif dataset == 'bank_marketing':
        return _load_bank_marketing()
    
    # Regression datasets (synthetic approximations)
    elif dataset == 'boston_housing':
        return _load_boston_housing()
    elif dataset == 'diabetes':
        return _load_diabetes()
    elif dataset == 'california_housing':
        return _load_california_housing()
    elif dataset == 'energy':
        return _load_energy()
    
    # Image datasets (synthetic approximations)
    elif dataset == 'mnist':
        return _load_mnist()
    elif dataset == 'fashion_mnist':
        return _load_fashion_mnist()
    elif dataset == 'cifar10' or dataset == 'cifar_10':
        return _load_cifar10()
    elif dataset == 'cifar100' or dataset == 'cifar_100':
        return _load_cifar100()
    elif dataset == 'imagenet':
        return _load_imagenet()
    
    # Text datasets (synthetic approximations)
    elif dataset == 'imdb' or dataset == 'imdb_reviews':
        return _load_imdb()
    elif dataset == 'penn_treebank' or dataset == 'ptb':
        return _load_penn_treebank()
    
    # Spiking datasets (synthetic approximations)
    elif dataset == 'ssc' or dataset == 'spiking_speech_commands':
        return _load_ssc()
    elif dataset == 'shd' or dataset == 'spiking_heidelberg_digits':
        return _load_shd()
    
    # Synthetic datasets
    elif dataset == 'synthetic' or dataset == 'synthetic_classification':
        n_samples = kwargs.get('n_samples', 100)
        n_features = kwargs.get('n_features', 4)
        n_classes = kwargs.get('n_classes', 2)
        return _make_classification(n_samples, n_features, n_classes)
    elif dataset == 'synthetic_reg' or dataset == 'synthetic_regression':
        n_samples = kwargs.get('n_samples', 100)
        n_features = kwargs.get('n_features', 4)
        return _make_regression(n_samples, n_features)
    
    else:
        raise ValueError(f"Unknown dataset: '{dataset}'. Available: iris, wine, breast_cancer, "
                        f"adult, heart_disease, bank_marketing, boston_housing, diabetes, "
                        f"california_housing, energy, mnist, fashion_mnist, cifar10, cifar100, "
                        f"imagenet, imdb, penn_treebank, ssc, shd, synthetic, synthetic_reg")


def _load_iris() -> Tuple[np.ndarray, np.ndarray]:
    """Load Iris dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 150
    n_features = 4
    n_classes = 3
    
    # Generate synthetic iris-like data
    X = np.random.randn(n_samples, n_features)
    X[:50] += np.array([5.1, 3.5, 1.4, 0.2])  # Setosa-like
    X[50:100] += np.array([5.9, 3.0, 4.2, 1.3])  # Versicolor-like
    X[100:] += np.array([6.6, 3.0, 5.6, 2.0])  # Virginica-like
    
    y = np.array([0] * 50 + [1] * 50 + [2] * 50)
    
    return X, y


def _load_wine() -> Tuple[np.ndarray, np.ndarray]:
    """Load Wine dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 178
    n_features = 13
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features) * 0.5
    X[:59] += np.array([13.7, 2.8, 2.6, 17.4, 106, 2.9, 3.0, 0.3, 2.3, 6.2, 1.1, 0.9, 570]) * 0.1
    X[59:130] += np.array([12.3, 2.8, 2.7, 21.5, 96, 2.0, 2.1, 0.3, 1.8, 4.0, 0.9, 0.8, 560]) * 0.1
    X[130:] += np.array([13.2, 3.2, 2.4, 19.5, 104, 2.0, 1.8, 0.3, 2.0, 3.5, 1.0, 0.8, 750]) * 0.1
    
    y = np.array([0] * 59 + [1] * 71 + [2] * 48)
    
    return X, y


def _load_breast_cancer() -> Tuple[np.ndarray, np.ndarray]:
    """Load Breast Cancer dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 569
    n_features = 30
    n_classes = 2
    
    X = np.random.randn(n_samples, n_features) * 0.5
    X[:212] += np.array([17.9, 10.4, 122.8, 1001, 0.12, 0.28, 0.30, 0.21, 0.26, 0.16,
                         0.66, 0.73, 0.60, 0.49, 0.28, 0.21, 0.38, 0.63, 0.75, 0.12,
                         0.34, 0.42, 2.87, 21.9, 0.71, 1.00, 1.26, 0.20, 0.40, 0.10]) * 0.05
    X[212:] += np.array([20.6, 17.9, 132.7, 1276, 0.09, 0.16, 0.20, 0.13, 0.21, 0.06,
                         0.25, 0.37, 0.28, 0.20, 0.14, 0.08, 0.21, 0.39, 0.49, 0.09,
                         0.18, 0.24, 1.73, 20.2, 0.57, 0.67, 0.73, 0.12, 0.26, 0.08]) * 0.05
    
    y = np.array([0] * 212 + [1] * 357)
    
    return X, y


def _load_boston_housing() -> Tuple[np.ndarray, np.ndarray]:
    """Load Boston Housing dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 506
    n_features = 13
    
    X = np.random.randn(n_samples, n_features) * 0.5
    X += np.array([0.01, 18.0, 2.3, 0.5, 0.5, 6.0, 65.0, 4.0, 1.0, 296.0, 15.0, 396.0, 5.0]) * 0.1
    
    # Generate target based on features
    y = (X.sum(axis=1) * 5 + np.random.randn(n_samples) * 5).clip(5, 50)
    
    return X, y


def _load_diabetes() -> Tuple[np.ndarray, np.ndarray]:
    """Load Diabetes dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 442
    n_features = 10
    
    X = np.random.randn(n_samples, n_features) * 0.1
    X += np.array([0.04, -0.05, 0.03, -0.03, -0.04, -0.03, 0.09, 0.03, -0.01, -0.04]) * 10
    
    y = (X.sum(axis=1) * 30 + 150 + np.random.randn(n_samples) * 20).clip(25, 350)
    
    return X, y


def _load_california_housing() -> Tuple[np.ndarray, np.ndarray]:
    """Load California Housing dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 20640
    n_features = 8
    
    X = np.random.randn(n_samples, n_features) * 0.5
    X += np.array([3.0, 25.0, 5.0, 1.2, 1400.0, 3.0, 35.0, -119.0]) * 0.1
    
    y = (X.sum(axis=1) * 10 + np.random.randn(n_samples) * 10).clip(0.15, 5.0)
    
    return X, y


def _load_energy() -> Tuple[np.ndarray, np.ndarray]:
    """Load Energy Efficiency dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 768
    n_features = 8
    
    X = np.random.randn(n_samples, n_features) * 0.5
    X += np.array([0.76, 642.5, 318.5, 0.1, 2.0, 0.5, 0.5, 3.0]) * 0.1
    
    y = (X.sum(axis=1) * 5 + np.random.randn(n_samples) * 2).clip(10, 50)
    
    return X, y


def _make_classification(n_samples: int = 100, n_features: int = 4, n_classes: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification data."""
    X = np.random.randn(n_samples, n_features)
    
    # Create class-separated data
    samples_per_class = n_samples // n_classes
    for i in range(n_classes):
        start_idx = i * samples_per_class
        end_idx = (i + 1) * samples_per_class if i < n_classes - 1 else n_samples
        X[start_idx:end_idx] += np.random.randn(1, n_features) * 2
    
    y = np.repeat(np.arange(n_classes), samples_per_class)
    if len(y) < n_samples:
        y = np.concatenate([y, np.array([n_classes - 1] * (n_samples - len(y)))])
    
    return X, y


def _make_regression(n_samples: int = 100, n_features: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data."""
    X = np.random.randn(n_samples, n_features)
    y = X.sum(axis=1) + np.random.randn(n_samples) * 0.5
    
    return X, y


def _load_adult() -> Tuple[np.ndarray, np.ndarray]:
    """Load Adult/Census Income dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 48842
    n_features = 14
    n_classes = 2
    
    X = np.random.randn(n_samples, n_features) * 0.3
    # Add realistic feature patterns
    X[:, 0] = np.random.randint(17, 90, n_samples)  # age
    X[:, 1] = np.random.randint(1, 16, n_samples)  # workclass
    X[:, 2] = np.random.randint(1, 20, n_samples) * 10000  # fnlwgt
    X[:, 3] = np.random.randint(1, 17, n_samples)  # education
    X[:, 4] = np.random.randint(1, 17, n_samples)  # education_num
    X[:, 5] = np.random.randint(1, 7, n_samples)  # marital_status
    X[:, 6] = np.random.randint(1, 15, n_samples)  # occupation
    X[:, 7] = np.random.randint(1, 7, n_samples)  # relationship
    X[:, 8] = np.random.randint(1, 3, n_samples)  # race
    X[:, 9] = np.random.randint(0, 2, n_samples)  # sex
    X[:, 10] = np.random.randint(1, 100, n_samples)  # capital_gain
    X[:, 11] = np.random.randint(0, 100, n_samples)  # capital_loss
    X[:, 12] = np.random.randint(1, 100, n_samples)  # hours_per_week
    X[:, 13] = np.random.randint(1, 60, n_samples)  # native_country
    
    y = np.random.randint(0, 2, n_samples)
    
    return X, y


def _load_heart_disease() -> Tuple[np.ndarray, np.ndarray]:
    """Load Heart Disease dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 303
    n_features = 13
    n_classes = 2
    
    X = np.random.randn(n_samples, n_features) * 0.5
    X += np.array([54.4, 0.68, 0.97, 130.0, 246.0, 0.15, 0.58, 1.4, 150.0, 0.28, 0.5, 1.0, 2.7]) * 0.1
    
    y = np.random.randint(0, 2, n_samples)
    
    return X, y


def _load_bank_marketing() -> Tuple[np.ndarray, np.ndarray]:
    """Load Bank Marketing dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 45211
    n_features = 16
    n_classes = 2
    
    X = np.random.randn(n_samples, n_features) * 0.3
    X[:, 0] = np.random.randint(18, 95, n_samples)  # age
    X[:, 1] = np.random.randint(1, 12, n_samples)  # job
    X[:, 2] = np.random.randint(1, 5, n_samples)  # marital
    X[:, 3] = np.random.randint(1, 5, n_samples)  # education
    X[:, 4] = np.random.randint(0, 2, n_samples)  # default
    X[:, 5] = np.random.randint(0, 2, n_samples)  # housing
    X[:, 6] = np.random.randint(0, 2, n_samples)  # loan
    X[:, 7] = np.random.randint(1, 10, n_samples)  # contact
    X[:, 8] = np.random.randint(1, 32, n_samples)  # month
    X[:, 9] = np.random.randint(1, 8, n_samples)  # day_of_week
    X[:, 10] = np.random.randint(1, 1000, n_samples)  # duration
    X[:, 11] = np.random.randint(1, 10, n_samples)  # campaign
    X[:, 12] = np.random.randint(-1, 100, n_samples)  # pdays
    X[:, 13] = np.random.randint(0, 100, n_samples)  # previous
    X[:, 14] = np.random.randint(1, 5, n_samples)  # poutcome
    X[:, 15] = np.random.uniform(0, 100, n_samples)  # emp_var_rate
    
    y = np.random.randint(0, 2, n_samples)
    
    return X, y


def _load_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 70000
    n_features = 784  # 28x28 flattened
    n_classes = 10
    
    X = np.random.randn(n_samples, n_features) * 0.5
    # Add digit-like patterns
    for i in range(n_classes):
        start_idx = i * (n_samples // n_classes)
        end_idx = (i + 1) * (n_samples // n_classes) if i < n_classes - 1 else n_samples
        X[start_idx:end_idx] += np.random.randn(1, n_features) * 0.3
    
    y = np.repeat(np.arange(n_classes), n_samples // n_classes)
    if len(y) < n_samples:
        y = np.concatenate([y, np.array([n_classes - 1] * (n_samples - len(y)))])
    
    return X, y


def _load_fashion_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """Load Fashion MNIST dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 70000
    n_features = 784
    n_classes = 10
    
    X = np.random.randn(n_samples, n_features) * 0.5
    for i in range(n_classes):
        start_idx = i * (n_samples // n_classes)
        end_idx = (i + 1) * (n_samples // n_classes) if i < n_classes - 1 else n_samples
        X[start_idx:end_idx] += np.random.randn(1, n_features) * 0.4
    
    y = np.repeat(np.arange(n_classes), n_samples // n_classes)
    if len(y) < n_samples:
        y = np.concatenate([y, np.array([n_classes - 1] * (n_samples - len(y)))])
    
    return X, y


def _load_cifar10() -> Tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 60000
    n_features = 3072  # 32x32x3 flattened
    n_classes = 10
    
    X = np.random.randn(n_samples, n_features) * 0.5
    for i in range(n_classes):
        start_idx = i * (n_samples // n_classes)
        end_idx = (i + 1) * (n_samples // n_classes) if i < n_classes - 1 else n_samples
        X[start_idx:end_idx] += np.random.randn(1, n_features) * 0.3
    
    y = np.repeat(np.arange(n_classes), n_samples // n_classes)
    if len(y) < n_samples:
        y = np.concatenate([y, np.array([n_classes - 1] * (n_samples - len(y)))])
    
    return X, y


def _load_cifar100() -> Tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-100 dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 60000
    n_features = 3072
    n_classes = 100
    
    X = np.random.randn(n_samples, n_features) * 0.5
    samples_per_class = n_samples // n_classes
    for i in range(n_classes):
        start_idx = i * samples_per_class
        end_idx = (i + 1) * samples_per_class if i < n_classes - 1 else n_samples
        X[start_idx:end_idx] += np.random.randn(1, n_features) * 0.2
    
    y = np.repeat(np.arange(n_classes), samples_per_class)
    if len(y) < n_samples:
        y = np.concatenate([y, np.array([n_classes - 1] * (n_samples - len(y)))])
    
    return X, y


def _load_imagenet() -> Tuple[np.ndarray, np.ndarray]:
    """Load ImageNet dataset (synthetic approximation - small sample)."""
    np.random.seed(42)
    n_samples = 10000  # Small sample for practicality
    n_features = 224 * 224 * 3  # 224x224x3 flattened
    n_classes = 1000
    
    X = np.random.randn(n_samples, n_features) * 0.5
    samples_per_class = n_samples // n_classes
    for i in range(min(n_classes, n_samples)):
        start_idx = i * samples_per_class
        end_idx = (i + 1) * samples_per_class if i < n_classes - 1 else n_samples
        if start_idx < n_samples:
            X[start_idx:end_idx] += np.random.randn(1, n_features) * 0.2
    
    y = np.repeat(np.arange(min(n_classes, n_samples)), samples_per_class)
    if len(y) < n_samples:
        y = np.concatenate([y, np.array([min(n_classes, n_samples) - 1] * (n_samples - len(y)))])
    
    return X, y


def _load_imdb() -> Tuple[np.ndarray, np.ndarray]:
    """Load IMDB Reviews dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 50000
    n_features = 5000  # Vocabulary size
    n_classes = 2
    
    X = np.random.randint(0, 2, (n_samples, n_features)).astype(float)
    # Add some structure
    for i in range(n_classes):
        start_idx = i * (n_samples // n_classes)
        end_idx = (i + 1) * (n_samples // n_classes) if i < n_classes - 1 else n_samples
        X[start_idx:end_idx, :100] = np.random.rand(end_idx - start_idx, 100) * 0.5
    
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    
    return X, y


def _load_penn_treebank() -> Tuple[np.ndarray, np.ndarray]:
    """Load Penn Treebank dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 42068
    n_features = 10000  # Vocabulary size
    
    X = np.random.randint(0, n_features, (n_samples, 50)).astype(float)
    y = np.random.randint(0, n_features, n_samples)
    
    return X, y


def _load_ssc() -> Tuple[np.ndarray, np.ndarray]:
    """Load Spiking Speech Commands dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 105829
    n_features = 700  # Time steps * channels
    n_classes = 35
    
    X = np.random.rand(n_samples, n_features) * 0.1
    samples_per_class = n_samples // n_classes
    for i in range(n_classes):
        start_idx = i * samples_per_class
        end_idx = (i + 1) * samples_per_class if i < n_classes - 1 else n_samples
        X[start_idx:end_idx] += np.random.rand(end_idx - start_idx, n_features) * 0.05
    
    y = np.repeat(np.arange(n_classes), samples_per_class)
    if len(y) < n_samples:
        y = np.concatenate([y, np.array([n_classes - 1] * (n_samples - len(y)))])
    
    return X, y


def _load_shd() -> Tuple[np.ndarray, np.ndarray]:
    """Load Spiking Heidelberg Digits dataset (synthetic approximation)."""
    np.random.seed(42)
    n_samples = 20282
    n_features = 256  # Time steps * channels
    n_classes = 20
    
    X = np.random.rand(n_samples, n_features) * 0.1
    samples_per_class = n_samples // n_classes
    for i in range(n_classes):
        start_idx = i * samples_per_class
        end_idx = (i + 1) * samples_per_class if i < n_classes - 1 else n_samples
        X[start_idx:end_idx] += np.random.rand(end_idx - start_idx, n_features) * 0.05
    
    y = np.repeat(np.arange(n_classes), samples_per_class)
    if len(y) < n_samples:
        y = np.concatenate([y, np.array([n_classes - 1] * (n_samples - len(y)))])
    
    return X, y


def load_synthetic_cnn_data(
    n_samples: int = 1000,
    height: int = 28,
    width: int = 28,
    channels: int = 1,
    classes: int = 10,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic image data for CNN examples.
    
    Args:
        n_samples: Number of samples to generate
        height: Image height
        width: Image width
        channels: Number of channels (1 for grayscale, 3 for RGB)
        classes: Number of classes
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y) where X has shape (n_samples, height, width, channels)
        and y has shape (n_samples,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random image data
    X = np.random.randn(n_samples, height, width, channels).astype(np.float32)
    
    # Assign labels based on pixel sum thresholds
    pixel_sums = X.sum(axis=(1, 2, 3))
    thresholds = np.percentile(pixel_sums, np.linspace(0, 100, classes + 1)[1:-1])
    y = np.digitize(pixel_sums, thresholds)
    
    return X, y


def load_synthetic_snn_data(
    n_samples: int = 1000,
    input_size: int = 10,
    time_steps: int = 20,
    classes: int = 2,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic spiking data for SNN examples.
    
    Args:
        n_samples: Number of samples to generate
        input_size: Number of input neurons/features
        time_steps: Number of time steps
        classes: Number of classes
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y) where X has shape (n_samples, input_size) and y has shape (n_samples,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random input features
    X = np.random.randn(n_samples, input_size).astype(np.float32)
    
    # Assign labels based on feature sum thresholds
    feature_sums = X.sum(axis=1)
    thresholds = np.percentile(feature_sums, np.linspace(0, 100, classes + 1)[1:-1])
    y = np.digitize(feature_sums, thresholds)
    
    return X, y


def load_synthetic_graph_data(
    n_nodes: int = 100,
    n_features: int = 16,
    n_classes: int = 4,
    edge_prob: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic graph data for GNN examples.
    
    Args:
        n_nodes: Number of nodes in the graph
        n_features: Number of features per node
        n_classes: Number of classes
        edge_prob: Probability of edge between nodes
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (node_features, adjacency_matrix, labels)
        node_features: (n_nodes, n_features)
        adjacency_matrix: (n_nodes, n_nodes)
        labels: (n_nodes,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random node features
    node_features = np.random.randn(n_nodes, n_features).astype(np.float32)
    
    # Generate random adjacency matrix
    adjacency_matrix = np.random.rand(n_nodes, n_nodes) < edge_prob
    adjacency_matrix = adjacency_matrix.astype(np.float32)
    np.fill_diagonal(adjacency_matrix, 0)  # No self-loops
    
    # Assign labels based on feature sum thresholds
    feature_sums = node_features.sum(axis=1)
    thresholds = np.percentile(feature_sums, np.linspace(0, 100, n_classes + 1)[1:-1])
    labels = np.digitize(feature_sums, thresholds)
    
    return node_features, adjacency_matrix, labels


def load_synthetic_sequence_data(
    n_samples: int = 1000,
    vocab_size: int = 1000,
    max_seq_len: int = 32,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic sequence data for Transformer examples.
    
    Args:
        n_samples: Number of samples to generate
        vocab_size: Size of vocabulary
        max_seq_len: Maximum sequence length
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y) where X and y have shape (n_samples, max_seq_len)
        containing integer token IDs
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random sequences
    X = np.random.randint(0, vocab_size, (n_samples, max_seq_len)).astype(np.int32)
    y = np.random.randint(0, vocab_size, (n_samples, max_seq_len)).astype(np.int32)
    
    return X, y
