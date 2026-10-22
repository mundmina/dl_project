"""Classical baseline: HOG features + linear SVM, trained on original images only.

HOG gives 8100 features for ~3k training images, so a linear SVM on raw features separates
the training set perfectly and overfits (and is slow to converge for larger C). PCA to a few
hundred components before the SVM fixes both.
"""
import os

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

C_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
PCA_COMPONENTS = 256


def hog_features(img, size=128):
    gray = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255
    return hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               feature_vector=True).astype(np.float32)


def _chunk(dataset, idxs, size):
    return np.stack([hog_features(dataset.render(i), size) for i in idxs])


def extract(dataset, size=128):
    """HOG for every item of a PetBackgroundDataset (any condition), in parallel on all CPUs."""
    n_jobs = os.cpu_count()
    chunks = np.array_split(np.arange(len(dataset)), 4 * n_jobs)
    parts = Parallel(n_jobs=n_jobs)(delayed(_chunk)(dataset, c, size) for c in chunks if len(c))
    return np.concatenate(parts)


def make_svm(C, n_components=PCA_COMPONENTS):
    return make_pipeline(StandardScaler(), PCA(n_components, random_state=0),
                         LinearSVC(C=C, dual="auto", max_iter=5000))
