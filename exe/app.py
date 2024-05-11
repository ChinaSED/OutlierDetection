import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
# Import other models from PyOD as needed
# Import all models

from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP
from pyod.models.inne import INNE
from pyod.models.gmm import GMM
from pyod.models.kde import KDE
from pyod.models.lmdd import LMDD

from pyod.models.dif import DIF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.suod import SUOD
from pyod.models.qmcd import QMCD
from pyod.models.sampling import Sampling
from pyod.models.kpca import KPCA
from pyod.models.lunar import LUNAR

detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
				 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
				 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
				 LOF(n_neighbors=50)]


@st.cache_data
def generate_data(n_samples, outliers_fraction, clusters_separation):
    np.random.seed(42)
    n_inliers = int((1. - outliers_fraction) * n_samples)
    n_outliers = int(outliers_fraction * n_samples)
    X1 = 0.3 * np.random.randn(n_inliers // 2, 2) - clusters_separation
    X2 = 0.3 * np.random.randn(n_inliers // 2, 2) + clusters_separation
    X = np.r_[X1, X2]
    X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]
    return X

def visualize_model(clf, X, outliers_fraction):
    clf.fit(X)
    scores_pred = clf.decision_function(X) * -1
    threshold = np.percentile(scores_pred, 100 * outliers_fraction)
    n_inliers = int((1. - outliers_fraction) * X.shape[0])
    n_outliers = X.shape[0] - n_inliers
    xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Blues_r)
    ax.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')
    ax.scatter(X[:n_inliers, 0], X[:n_inliers, 1], c='white', s=20, edgecolor='k')
    ax.scatter(X[n_inliers:, 0], X[n_inliers:, 1], c='black', s=20, edgecolor='k')
    ax.set_title(clf.__class__.__name__)
    ax.set_xlim((-7, 7))
    ax.set_ylim((-7, 7))
    st.pyplot(fig)

st.title("离群点检测")

n_samples = st.sidebar.slider("样本数量", 100, 1000, 200)
outliers_fraction = st.sidebar.slider("离群点占比", 0.0, 0.5, 0.25)
clusters_separation = st.sidebar.slider("样本集中度", -5.0, 5.0, 0.0)
random_state = 42

X = generate_data(n_samples, outliers_fraction, clusters_separation)
classifiers = {
	'Angle-based Outlier Detector (ABOD)':
		ABOD(contamination=outliers_fraction),
	'K Nearest Neighbors (KNN)': KNN(
		contamination=outliers_fraction),
	'Average KNN': KNN(method='mean',
					   contamination=outliers_fraction),
	'Median KNN': KNN(method='median',
					  contamination=outliers_fraction),
	'Local Outlier Factor (LOF)':
		LOF(n_neighbors=35, contamination=outliers_fraction),

	'Isolation Forest': IForest(contamination=outliers_fraction,
								random_state=random_state),
	# 'Deep Isolation Forest (DIF)': DIF(contamination=outliers_fraction,
	# 								   random_state=random_state),
	'INNE': INNE(
		max_samples=2, contamination=outliers_fraction,
		random_state=random_state,
	),

	'Locally Selective Combination (LSCP)': LSCP(
		detector_list, contamination=outliers_fraction,
		random_state=random_state),
	'Feature Bagging':
		FeatureBagging(LOF(n_neighbors=35),
					   contamination=outliers_fraction,
					   random_state=random_state),
	'SUOD': SUOD(contamination=outliers_fraction),

	'Minimum Covariance Determinant (MCD)': MCD(
		contamination=outliers_fraction, random_state=random_state),

	'Principal Component Analysis (PCA)': PCA(
		contamination=outliers_fraction, random_state=random_state),
	'KPCA': KPCA(
		contamination=outliers_fraction),

	'Probabilistic Mixture Modeling (GMM)': GMM(contamination=outliers_fraction,
												random_state=random_state),

	'LMDD': LMDD(contamination=outliers_fraction,
				 random_state=random_state),

	'Histogram-based Outlier Detection (HBOS)': HBOS(
		contamination=outliers_fraction),

	'Copula-base Outlier Detection (COPOD)': COPOD(
		contamination=outliers_fraction),

	'ECDF-baseD Outlier Detection (ECOD)': ECOD(
		contamination=outliers_fraction),
	'Kernel Density Functions (KDE)': KDE(contamination=outliers_fraction),

	'QMCD': QMCD(
		contamination=outliers_fraction),

	'Sampling': Sampling(
		contamination=outliers_fraction),

	'LUNAR': LUNAR(),

	'Cluster-based Local Outlier Factor (CBLOF)':
		CBLOF(contamination=outliers_fraction,
			  check_estimator=False, random_state=random_state),

	'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
}
CHANGE = False
def flag():
    CHANGE = True

algorithm = st.sidebar.selectbox("算法选择", classifiers.keys(), on_change=flag)
clf = classifiers[algorithm]

if st.sidebar.button("可视化") or CHANGE: ...
visualize_model(clf, X, outliers_fraction)