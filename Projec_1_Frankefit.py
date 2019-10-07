import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn import linear_model

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y#[(window_len/2-1):-(window_len/2)]


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4


def CreateDesignMatrix_X(x, y, n = 5):
	# @Author Morten Hjort-Jensen
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
	This is a vestigial function, used during debugging.
	It is not in use in the current version of the code
	"""
	deg = int((-3+np.sqrt(9-8*(1-len(beta))))/2)+1
	z = 0
	for i in range(deg):

		q = int((i)*(i+1)/2)

		for k in range(i+1):
			z += beta[q+k]*x**(i-k) * y**k

	return z


def MSE(y,ytilde):
	return 1./len(y)*sum((y-ytilde)**2)

def R2(y,ytilde):
	num = sum((y-ytilde)**2)
	den = sum((y-np.ones(len(y))*np.mean(y))**2)
	return 1-(num/den)

def kfold(x,y,k,deg=2,model='OLS',l=0):



	kf = KFold(n_splits=k, shuffle=True, random_state=2)
	betas = []
	train_mse = []
	for train_index, test_index in kf.split(x):

		x_train_k, x_val = x[train_index], x[test_index]
		y_train_k, y_val = y[train_index], y[test_index]

		z_train_k = FrankeFunction(x_train_k,y_train_k)
		z_train_k += np.random.normal(scale = stdev,size=np.shape(z_train_k))
		z_val = FrankeFunction(x_val,y_val)
		z_val += np.random.normal(scale = stdev,size=np.shape(z_val))

		X = CreateDesignMatrix_X(x_train_k,y_train_k,deg)
		X_i = X.T.dot(X)

		if model == 'OLS':
			beta = (np.linalg.inv(X_i) @ X.T ) @ z_train_k.ravel()

		elif model == 'Ridge':
			X_i = X_i + np.identity(np.shape(X_i)[0]) * l
			beta = (np.linalg.inv(X_i) @ X.T ) @ z_train_k.ravel()

		elif model == 'Lasso':
			clf = linear_model.Lasso(alpha=l,max_iter = 2000)
			clf.fit(X,z_train_k.ravel())

			beta = clf.coef_
			beta[0] = clf.intercept_


		pred = X @ beta
		train_mse.append(np.mean((z_train_k.ravel()-pred)**2))
		betas.append(beta)


	return betas, np.mean(train_mse)
		
from sklearn.model_selection import KFold, train_test_split

n_x=50  # number of points
m=15       # degree of polynomial
stdev = 0.2
k = 5    # number of folds
seed = 7


# INITIALIZE THE DATA:
x = np.sort(np.random.uniform(0, 1, n_x))
y = np.sort(np.random.uniform(0, 1, n_x))

x_train, x_test, y_train, y_test = train_test_split(x,y)

x, y = np.meshgrid(x,y)
x_train, y_train = np.meshgrid(x_train,y_train)
x_test, y_test = np.meshgrid(x_test,y_test)

z = FrankeFunction(x,y)
z_train = FrankeFunction(x_train,y_train)
np.random.seed(seed)
z_train = z_train + np.random.normal(scale = stdev,size=np.shape(z_train))
z_test = FrankeFunction(x_test,y_test)
np.random.seed(seed)
z_test = z_test + np.random.normal(scale = stdev,size=np.shape(z_test))



MSE = []
VAR = []
BIAS = []
train_MSE = []

lambdas = np.geomspace(10**-10, 0.001, 300, endpoint=False)

mse_matrix = np.zeros((len(lambdas),m))
#deg = 5
#for i,l in enumerate(lambdas):
	#print(i,'/',len(lambdas))
for j,deg in enumerate(range(m)):

	X_test = CreateDesignMatrix_X(x_test,y_test,deg)

	betas, train_mse = kfold(x_train,y_train,k,deg=deg,model = 'OLS',l=0.01)

	z_preds = np.zeros((np.shape(z_test)[0],np.shape(z_test)[1],k))
	mses = []
	trainmses = []

	for n, beta in enumerate(betas):
		z_preds[:,:,n] = (X_test@beta).reshape((np.shape(z_test)[0],np.shape(z_test)[1]))
		mses.append(np.mean((z_test-z_preds[:,:,n])**2))

	E_z = np.mean(z_preds,axis=2)
	var = np.var(z_preds,axis=2)
	bias = np.mean((z_test - E_z)**2)

	MSE.append(np.mean(mses))
	BIAS.append(bias)
	VAR.append(np.mean(var))
	train_MSE.append(train_mse)


plt.figure()
plt.plot(MSE,'-r',label = 'MSE')#,alpha=0.5)
plt.plot(train_MSE,'-b', label='Train MSE')
#plt.plot(lambdas,smooth(np.array(MSE),window_len=11)[:len(lambdas)],'--k')
#plt.plot(VAR,'--',label='Variance')
#plt.plot(BIAS,'--',label = 'Bias')
#plt.plot(np.array(VAR)+np.array(BIAS),'or',label='sum')
plt.gca().set_ylim([0,0.3])
plt.legend()
#plt.gca().set_ylim([0,3])

plt.show()
"""
plt.figure()
plt.imshow(mse_matrix)
plt.show()
"""