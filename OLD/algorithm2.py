#Accoding to Promoting Diversity in Recommendation 
#the Algrithim1 alpha>0,beta>0
# 
#Created by Sect(2014)

maxMovieId=3952
maxUserID=6040
rating_MAX=4.6
import math
import numpy as np

###################################################
'''
Input:
	pu:an matrix of dimension N x K from SVD
	qi:an matrix of dimension M x K from SVD
	user_movie: (user:movie:rating)
	movie_user: (movie:user:rating)
	u:userId
	k:the numbers of Recommendation	
	alpha:
	beta:
Output:
	the top-k Recommendation set S 
'''
#################################################

def maxInS(qi,S,j):
	max=0
	l=0
	med=0
	for indl in S:
		med=np.dot(qi[indl].reshape(1,-1),qi[j])[0][0]
		if med>max:
			l=indl
			max=med
	return max

def Max(a,b):
	if a>b:
		return a
	else:
		return b

def W(rating,j):
	return math.e**(rating[j]-rating_MAX)

def mmd(pu,qi,user_movie,movie_user,u,k,Alpha,Beta):

	S=set([])
	I=set(xrange(1,maxMovieId+1))
	Omega=set([])
	for index in user_movie[u]:
		Omega.add(index)
	Theta=I-Omega

	max=0
	w=0
	med=0
	for indi in Theta:
		if qi.has_key(indi) and len(movie_user[indi])>10:
			med=np.dot(pu[u].reshape(1,-1),qi[indi])[0][0]
			if med>max:
				w=indi
				max=med
	S.add(w)

	while len(S)<k:
		max=0
		w=0
		med=0
		for indi in Theta:
			if qi.has_key(indi) and len(movie_user[indi])>10:
				med=np.dot(pu[u].reshape(1,-1),qi[indi])[0][0]
				for indj in Omega:	
					med=med+0.5*Alpha*W(user_movie[u],indj)*Max(0,np.dot(qi[indi].reshape(1,-1),qi[indj])[0][0]-maxInS(qi,S,indj))
				for indj in S:
					med=med+0.5*Beta*np.linalg.norm(qi[indi]-qi[indj],2)
				if med>max:
					w=indi
					max=med
		S.add(w)
		Theta.remove(w)	
	return S
