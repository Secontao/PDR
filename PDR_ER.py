'''
Reference paper:
    "Promoting Diversity in Recommendation in Entropy Regularizer"
     
'''

maxMovieId=3952
maxUserID=6040
import copy

import math
import pickle
import numpy as np

def SIG_SS(SIG,S,sigmaE2):
	x=list(S)
	x=np.array(x)-1
	x=list(x)
	SIG_xx=(SIG[x,:][:,x]+sigmaE2*np.eye(len(S)))
	SIG_xx=np.mat(SIG_xx)
	SIG_xx=SIG_xx.I
	SIG_xx=np.array(SIG_xx)
	return SIG_xx

def ER(U,V,user_movie,movie_user,u,sigmaE2,sigmaU2,k,lambada):
	SIG=sigmaU2*np.dot(V,V.transpose())
	#I:
	#Oemga:
	#A:	
	I=set(xrange(1,maxMovieId+1))
	Omega=set([])
	A=set([])
	rOmega=[]
	for index in user_movie[u]:
		Omega.add(index)
		rOmega.append(user_movie[u][index])
	rOmega=np.array(rOmega)
	Theta=I-Omega
	S=set([])
	A=copy.deepcopy(Omega)

	D=SIG_SS(SIG,Omega,sigmaE2)
	
	for i in xrange(k):
		C=SIG_SS(SIG,A,sigmaE2)
		p_g={}
		p_R={}
		for w in Theta:
			if movie_user.has_key(w) and len(movie_user[w])>10:	
				a=list(A)
				a=np.array(a)-1
				a=list(a)	
				SIG_wA=SIG[w-1,a]
				p_g[w]=math.log(1*2*math.pi*math.e*(sigmaE2+np.dot(np.dot(SIG_wA,C),SIG_wA.transpose())))

				o=list(Omega)
				o=np.array(o)-1
				o=list(o)	
				SIG_wO=SIG[w-1,o]
				p_R[w]=np.dot(np.dot(SIG_wO,D),rOmega)
		max=-99999999
		w=0
		
		for j in Theta:
			if movie_user.has_key(j)and len(movie_user[j])>10:
				med=p_R[j]+lambada*p_g[j]
				if med>max:
					max=med
					w=j
		S.add(w)
		A.add(w)
		Theta.remove(w)
	return S

