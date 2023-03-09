
import sys, os
proj_path = os.path.dirname(__file__)+"/../../"
sys.path.append(proj_path)

#from .vmfmixtures import *
from vmfmixtures.src.vmfmixtures import *
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture


if __name__ == "__main__":
    np.random.seed(seed=100)
    #data = loadmat('./3d_data.mat')
    data = loadmat('./10d_data.mat')
    labels = data['z'].reshape(-1)
    #print(data)
    data = data['data']
    data = data.T
    #print(data.shape, data)

    # Clustering using vmf
    vmf = VMFMixture(data, 10, 10)
    vmf.run()
    print('elbo', vmf.ELBO)
    pred, scores = vmf.predict()
    unique, counts = np.unique(pred, return_counts=True)
    print('prediction:', dict(zip(unique, counts)))


    # Using GMM
    gmm = GaussianMixture(n_components=10, random_state=0).fit(data.T)
    pred_gmm = gmm.predict(data.T)
    unique, counts = np.unique(pred_gmm, return_counts=True)
    print('gmm prediction:', dict(zip(unique, counts)))

    # True clusters
    true_unique, true_counts = np.unique(labels, return_counts=True)
    print('true:', dict(zip(true_unique, true_counts)))
