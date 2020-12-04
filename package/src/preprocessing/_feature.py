import numpy as np


"""@Author Moad Taoufik"""
class FeatureSelector:
    #Class used to select features
    def __init__(self, output_dimension: int = None, variance_treshhold: float = 0.95):
        if variance_treshhold > 1:
            raise Exception("variance_treshhold must be less than 1")
        self.output_dimension = output_dimension
        self.variance_treshhold = variance_treshhold #variance_explained kept in percent
        self.eigenvectors = None
        self.eigenvalues = None

    def fit(self, X_df):
        #Takes pandas X dataframe as input and prepares for PCA transformation
        X = X_df.to_numpy()
        cov = np.cov(X, rowvar=False) #covariance matrix of our samples
        u, s, _ = np.linalg.svd(cov) #apply singular value decomposition to get eigenvalues and vectors of covariance matrix
        self.eigenvalues = s #store results
        self.eigenvectors = u
        print(len(s), "componenents (PCA)")


    def transform(self, X_df):
        X = X_df.to_numpy()
        if self.output_dimension == None: #if user does not provide output dimension use variance_treshhold as a criterion to apply pca
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


    

