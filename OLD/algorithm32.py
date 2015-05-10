maxMovieId=3952
maxUserID=6040

import numpy as np
import time


def mmd(pu,qi,user_movie,movie_user,u,k,Beta):
	S=set([])
	I=set(xrange(1,maxMovieId+1))
	Omega=set([])
	for index in user_movie[u]:
		Omega.add(index)
	Theta=I-Omega

	max=0
	w=0
	for indi in Theta:
		med=0
		if qi.has_key(indi) and len(movie_user[indi])>10:
			med=np.dot(pu[u].transpose(),qi[indi])[0][0]
			if med>max:
				w=indi
				max=med
	S.add(w)
	Theta.remove(w)
	
	while len(S)<k:
		max=0
		w=0
		med=0
		for indi in Theta:
			if qi.has_key(indi) and len(movie_user[indi])>10:
				med=np.dot(pu[u].transpose(),qi[indi])[0][0]
				med1=med
				for indj in S:
					med=med+Beta*np.linalg.norm(qi[indi]-qi[indj],2)
				if med>max:
					w=indi
					max=med
		S.add(w)
		Theta.remove(w)	
	return S	

