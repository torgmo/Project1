
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import KFold, train_test_split
import pydicom as dicom


# This piece of code is used to create the 2d exaples of over and underfitting.

def Kfold(x,y,k):

	kf = KFold(n_splits=k, shuffle=True, random_state=1)
	test_size = int(len(y)/k)
	y_pred = np.zeros((test_size,k))
	print(np.shape(y_pred))
	i = 0
	MSE = []
	betas = []
	varis = []


	for train_index, test_index in kf.split(x):
		
		
		#print("TRAIN:", len(train_index), "TEST:", len(test_index))
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]

		t1,t2 = zip( *sorted( zip(x_train, y_train) ) )
		x_train = np.array(t1)
		y_train = np.array(t2)

		X = designMatrix(x_train,m)
		X_i = np.linalg.inv(X.T.dot(X)).dot(X.T)
		beta = X_i.dot(y_train)
		
		#y_pred[:,i] = X @ beta
		#er = (y_test-y_pred[:,i])**2
		#mse = np.mean(er)
		#var = np.var(er)
		#print(i, np.shape(X),np.shape(X @ beta))

		#plt.plot(X[:,1],X @ beta,'--g')
		X_full = designMatrix(np.sort(np.linspace(0,1,100)),m)
		if i == 0:
			plt.plot(X_full[:,1],X_full @ beta,'--g',label = 'Fit')
		else:
			plt.plot(X_full[:,1],X_full @ beta,'--g')
			print(i)

		#MSE.append(mse)
		#varis.append(var)
		

		i+=1

	
	#return beta, np.mean(MSE),0,np.mean(varis)

	





def f(x):
	return 5*np.sin(0.5/x)
	#return 5+2*x-x**3+0.5*x**2

def designMatrix(x,m):
	X = np.ones((len(x),m+1))
	for i in range(m+1):
		X[:,i] = x**i

	return X


sigma = 0.5
x = np.linspace(0,1,20)
print(x[0])
y = f(x) + np.random.normal(scale= sigma, size=len(x))



from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.4)


t1,t2 = zip( *sorted( zip(x_train, y_train) ) )
x_train = np.array(t1)
y_train = np.array(t2)
print('HERE ',type(t1),type(t2))
MSE = []

plt.figure()
degs = [2,8]
for i,m in enumerate(degs):
	plt.subplot(2,1,i+1)
	plt.plot(x_train,y_train,'ob',label = 'Train set')
	plt.plot(x_test,y_test,'or',label = 'Test set')

	X = designMatrix(x_train,m)
	X_i = X.T.dot(X)
	beta = (np.linalg.inv(X_i) @ X.T ) @ y_train

	X_full = designMatrix(np.linspace(0,1,200),m)
	#plt.plot(X_full[:,1],X_full @ beta,'-g',label = 'fit')
	plt.plot(X_full[:,1],f(X_full[:,1]),'--k',label = 'True function')

	"""
	if i+1==1:
		plt.ylabel('Underfitting')
	if i+1==4 :
		plt.ylabel('Overfitting')
	plt.gca().set_ylim([-6,6])
	"""
	Kfold(x_train,y_train,5)
	plt.gca().set_ylim([-12,12])
	plt.legend()
	plt.title('Polynomial degree '+str(m))
	"""
	X_train = designMatrix(x_train,m)
	X_test = designMatrix(x_test,m)
	print(np.shape(X_train),np.shape(X_test))
	X_i = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T)
	beta = X_i.dot(y_train)
	print(m,beta)
	print(np.shape(beta),np.shape(X_test))		
	y_pred = beta.dot(X_test.T)

	MSE.append(np.mean((y_test-y_pred)**2))
	"""


plt.show()




