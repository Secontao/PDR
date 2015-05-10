maxMovieId=3952
maxUserID=6040
import copy
import math
import pickle
import numpy as np

def matrix(qi,S,sigmaE2,sigmaU2):
	count=0	
	VO=np.array([])
	for i in S:
		if count==0:
			VO=qi[i]
			count=count+1
		else:
			VO=np.hstack((VO,qi[i]))
	Sigma=sigmaU2*np.dot(VO.transpose(),VO)
	matri=(Sigma+sigmaE2*np.eye(len(S)))**-1
	return matri

def ER(pu,qi,user_movie,movie_user,u,sigmaE2,sigmaU2,k,lambada):
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

	D=matrix(qi,Omega,sigmaE2,sigmaU2)
	
	for j in xrange(k):
		C=matrix(qi,A,sigmaE2,sigmaU2)
		p_g={}
		p_R={}
		for i in Theta:
			if qi.has_key(i) and len(movie_user[i])>10:	
				Sigma_iA=[]
				for indi in A:
					Sigma_iA.append(np.dot(qi[i].transpose(),qi[indi])[0][0])
				Sigma_iA=np.array(Sigma_iA)
				p_g[i]=0.5*math.log(2*math.pi*math.e*(sigmaE2+np.dot(np.dot(Sigma_iA,C),Sigma_iA.transpose())))

				Sigma_iO=[]
				for indi in Omega:
					Sigma_iO.append(np.dot(qi[i].transpose(),qi[indi])[0][0])
				Sigma_iO=np.array(Sigma_iO)
				p_R[i]=np.dot(np.dot(Sigma_iO,D),rOmega)
		max=0
		i=0
		med=0
		for indi in Theta:
			if qi.has_key(indi)and len(movie_user[indi])>10:
				med=p_R[indi]+lambada*p_g[indi]
				if med>max:
					max=med
					i=indi
		S.add(i)
		A.add(i)
		Theta.remove(i)
	return S

