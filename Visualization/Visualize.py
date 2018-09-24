from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.io as scio

# feature = np.load("resnet.npy")
# feature = pickle.load(open('vgg.p', 'rb')
mat = scio.loadmat('scatnet_2.mat')
feature = mat['b'].transpose()
feature = feature[0:2000]
print(feature.shape)
label = np.load("label.npy")
label = label[0:2000]
color = ['r','g','b','c','y','m','k','#ef5060','#5060ef','#50ef60']

x_pca = PCA(n_components=2).fit_transform(feature)
print('pca')
plt.figure(0)
for i in range(10):
    plt.scatter(x_pca[label==i,0],x_pca[label==i,1],c=color[i])
plt.savefig("scatnet_2_pca.png")

x_tnse = TSNE(n_components=2).fit_transform(feature)
print('tnse')
plt.figure(1)
for i in range(10):
    plt.scatter(x_tnse[label==i,0],x_tnse[label==i,1],c=color[i])
plt.savefig("scatnet_2_tsne.png")

x_mds = MDS(n_components=2).fit_transform(feature)
print('MDS')
feature = feature.astype(np.float64)
plt.figure(2)
for i in range(10):
    plt.scatter(x_mds[label==i,0],x_mds[label==i,1],c=color[i])
plt.savefig("scatnet_2_mds.png")

plt.show()
