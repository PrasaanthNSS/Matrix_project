import numpy as np
import matplotlib.pyplot as plt
import math

len=49
h=0.000001
t=np.linspace(0,2*np.pi,len)

E=np.zeros((2,len))
T_vec=np.zeros((2,len))
A=np.zeros((2,len))
B=np.zeros((2,len))

def x(z):
	return (np.sqrt(17.))*(np.cos(z))
	
def y(z):
	return (np.sqrt(17.)/3.)*(np.sin(z))

def xp(z):
	return (x(z+h)-x(z))/h

def yp(z):
	return (y(z+h)-y(z))/h 	
		
for i in range(len):
	temp_1=np.array([x(t[i]),y(t[i])])
	E[:,i]=temp_1.T
	temp_2=np.array([xp(t[i]),yp(t[i])])
	T_vec[:,i]=temp_2.T

Xaxis=np.array([[0,1]])
Yaxis=np.array([[1,0]])
	

def line_intersect(E,CF):
	n1=np.matmul(np.array([[0,1],[-1,0]]),T_vec[:,i:i+1])
	n2=CF
	N=np.vstack((n1.T,n2))
	p=np.zeros((2,1))
	p[0:1,:]=np.matmul(n1.T,E[:,i:i+1])
	p[1:,:]=0
	return np.matmul(np.linalg.inv(N),p)

X=np.zeros((2,len))
Y=np.zeros((2,len))
for i in range(len):
	X[:,i:i+1]=line_intersect(E,Xaxis)
	Y[:,i:i+1]=line_intersect(E,Yaxis)	

AREA=np.zeros(len)
for i in range(len):
	AREA[i]=abs(X[0:1,i:i+1]*Y[1:,i:i+1]/2)
MIN_AREA=100
for i in range(len):
	if (MIN_AREA >= AREA[i]):
		MIN_AREA=AREA[i]
	
logAREA=np.log10(AREA)	

print(MIN_AREA)
#print(t)
z=np.linspace(-7.0,7.0,2)
x=0*z
L=1.943*(1-z*0.171)
Lx=np.linspace(-0.01,0.01,2)
y=600*Lx
plt.figure(1)
plt.plot(E[0,:],E[1,:],label='$AB$')
plt.plot(z,x,label='$Xaxis$')
plt.plot(z,L,label='$Tangent$')
plt.plot(Lx,y,label='$Yaxis$')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')


plt.figure(2)
plt.plot(t,logAREA,label='$logAREA$')
plt.xlabel('$theta(t)$')
plt.ylabel('$log of AREA$')
plt.legend(loc="upper right")
plt.grid()
plt.show()

	
	

	
