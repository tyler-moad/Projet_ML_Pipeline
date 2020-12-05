import numpy as np
from sklearn.feature_selection import SelectKBest,mutual_info_classif
 # class FeatureSelector:
 #    def __init__(self):
 #        self.fs = SelectKBest(score_func=mutual_info_classif, k=5)
 #
 #    def fit(X,y,n_features=8):
 #        if n_features:
 #            self.fs.set_params(k=n_features)
	#     # self.fs.fit(X, y)
 #
 #    def transform(X):
 #        return self.fs.transform(X)
class FeatureSelector:
    def __init__(self, output_dimension: int = None, variance_treshhold: float = 0.95):
        self.output_dimension = output_dimension
        self.variance_treshhold = variance_treshhold
        self.eigenvectors = None
        self.eigenvalues = None

    def pca_fit(self, X_df):
        X = X_df.to_numpy()
        cov = np.cov(X, rowvar=False)
        u, s, v = np.linalg.svd(cov)
        self.eigenvalues = s
        self.eigenvectors = u
        print(len(s), "componenents (PCA)")

    def pca_transform(self, X_df):
        X = X_df.to_numpy()
        if self.output_dimension == None:
            explained_variance = 0
            total = sum(self.eigenvalues)
            compteur = 0
            while explained_variance < self.variance_treshhold:
                explained_variance = explained_variance + self.eigenvalues[compteur] / total
                compteur = compteur + 1
            U_k = self.eigenvectors[:, :compteur]  # (n x k)
            X_reduced = np.dot(X, U_k)
            print("data reduced to %d dimensions" % (compteur))
            return X_reduced
        else:
            if np.shape(X)[1] < self.output_dimension:
                raise Exception("Output Dimension > Input Dimension")
            U_k = self.eigenvectors[:, :self.output_dimension]  # (n x k)
            X_reduced = np.dot(X, U_k)
            print("data reduced to %d dimensions" % (self.output_dimension))
            return X_reduced


    

