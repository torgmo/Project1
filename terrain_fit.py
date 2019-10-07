import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random
from skimage.filters import gaussian
from imageio import imread
from sklearn import linear_model


def FrankeFunction(x,y,z):

	xs = x.ravel()
	ys = y.ravel()
	zs = np.zeros((len(xs),len(ys)))
	for i,xi in enumerate(xs):
		for j,yj in enumerate(ys):
			#print(i,j,xi,yj)
			zs[i,j] = z[xi,yj]

	return zs


def CreateDesignMatrix_X(x, y, n = 5):
	"""
	Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
	"""
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		#print('order ',i,':')
		q = int((i)*(i+1)/2)
		#print(q)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) * y**k
			#print(i-k,k)

	return X


def PolynomialFunction(x,y,beta):
	"""
	zorder = beta[0]
	firstorder = beta[1]*x+beta[2]*y
	secondorder = beta[3]*x**2+beta[5]*y**2+beta[4]*x*y
	thirdorder = beta[6]*x**3+beta[9]*y**3+beta[7]*x**2*y+beta[8]*x*y**2
	fourthorder = beta[10]*x**4+beta[14]*y**4+beta[11]*x**3*y+beta[12]*x**2*y**2+beta[13]*x*y**3
	fifthorder = beta[15]*x**5+beta[20]*y**5+beta[16]*x**4*y+beta[17]*x**3*y**2+beta[18]*x**2*y**3+beta[19]*x*y**4
 
	test = [zorder,firstorder,secondorder,thirdorder,fourthorder,fifthorder]
	"""
	deg = int((-3+np.sqrt(9-8*(1-len(beta))))/2)+1
	z = 0
	for i in range(deg):
		#print('order ',i,':')
		q = int((i)*(i+1)/2)
		#print(q)
		#z = 0
		for k in range(i+1):
			z += beta[q+k]*x**(i-k) * y**k

	return z #zorder+firstorder+secondorder+thirdorder+fourthorder+fifthorder


def MSE(y,ytilde):
	return 1./len(y)*sum((y-ytilde)**2)

def R2(y,ytilde):
	num = sum((y-ytilde)**2)
	den = sum((y-np.ones(len(y))*np.mean(y))**2)
	return 1-(num/den)

def kfold(x,y,z,k,deg=2,model='OLS',l=0):

	kf = KFold(n_splits=k, shuffle=True, random_state=2)
	betas = []
	c = 0
	for train_index, test_index in kf.split(x):
		#print("TRAIN:", len(train_index), "TEST:", len(test_index))
		x_train_k, x_val = x[train_index], x[test_index]
		y_train_k, y_val = y[train_index], y[test_index]
		z_train_k, z_val = z[train_index], z[test_index]

		X = CreateDesignMatrix_X(x_train_k,y_train_k,deg)
		X_i = X.T.dot(X)

		if model == 'OLS':
			beta = (np.linalg.inv(X_i) @ X.T ) @ z_train_k.ravel()

		elif model == 'Ridge':
			X_i += np.identity(np.shape(X_i)[0]) * l
			beta = (np.linalg.inv(X_i) @ X.T ) @ z_train_k.ravel()

		elif model == 'Lasso':
			clf = linear_model.Lasso(alpha=l,max_iter = 10000)
			clf.fit(X,z_train_k.ravel())

			beta = clf.coef_
			beta[0] = clf.intercept_

		betas.append(beta)


	return betas
		
from sklearn.model_selection import KFold, train_test_split

n_x=600  # number of points
m=10     # maximum degree of polynomial

stdev = 0.2
k = 5   # number of folds
seed = 7

terrain = np.array(imread(r'C:\Users\Tiorgeir\Documents\FYS-STK\Project1\SRTM_data_Norway_1.tif'))
terrain = (terrain - np.mean(terrain))/np.var(terrain)
np.random.seed(seed)
x = np.sort(np.random.choice(np.shape(terrain)[0],n_x,replace=False))
np.random.seed(seed)
y = np.sort(np.random.choice(np.shape(terrain)[1],n_x,replace=False))
z = FrankeFunction(x,y,terrain)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.95)

z_train = FrankeFunction(x_train,y_train,terrain)
z_test = FrankeFunction(x_test,y_test,terrain)

del terrain # I need all my memory...

x, y = np.meshgrid(x,y)
x_train, y_train = np.meshgrid(x_train,y_train)
x_test, y_test = np.meshgrid(x_test,y_test)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x,y,z,cmap=cm.terrain)
ax.scatter(x_train,y_train,z_train,c='k',marker='x')
plt.show()

MSE = []
VAR = []
BIAS = []

deg = 5
l = 0.01
lambdas = [10**-9,10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1,10**0]# np.geomspace(10**-10, 0.001, 100, endpoint=False)
#lambdas = np.geomspace(10**-10, 0.001, 100, endpoint=False)

mod = 'OLS'
lambdas = [0]
mse_matrix = np.zeros((len(lambdas),m))
for d,deg in enumerate(range(m)):

	print('Deg ',deg,'/',m-1)
	for j,l in enumerate(lambdas):
			
			#print('lambda ',j,'/',len(lambdas))

		X_test = CreateDesignMatrix_X(x_test,y_test,deg)
		betas = kfold(x_train,y_train,z_train,k,deg=deg,model = mod,l=l)

		z_preds = np.zeros((np.shape(z_test)[0],np.shape(z_test)[1],k))

		mses = []
		trainmses = []

		for i, beta in enumerate(betas):
			z_preds[:,:,i] = (X_test@beta).reshape((np.shape(z_test)[0],np.shape(z_test)[1]))
			mses.append(np.mean((z_test-z_preds[:,:,i])**2))

		E_z = np.mean(z_preds,axis=2)
		var = np.var(z_preds,axis=2)
		bias = np.mean((z_test - E_z)**2)

		MSE.append(np.mean(mses))
		BIAS.append(bias)
		VAR.append(np.mean(var))

		mse_matrix[j,d] = np.mean(mses)

np.save(r'C:\Users\Tiorgeir\Documents\FYS-STK\Project1\MSE_matrix_'+mod,mse_matrix)

plt.figure()
plt.plot(MSE,label = 'MSE')
plt.plot(VAR,'--',label='Variance')
plt.plot(BIAS,'--',label = 'Bias')
plt.plot(np.array(VAR)+np.array(BIAS),'or',label='sum')
plt.legend()
plt.title(mod)
plt.gca().set_ylim([0,0.003])

plt.show()


"""
Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
"""