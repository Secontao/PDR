import pickle
import numpy as np
import time

t1=time.clock()

input2=open('qi.pkl')
input3=open('test_user_movie.pkl')

qi=pickle.load(input2)
test_user_movie=pickle.load(input3)

def novelty(u,i):
	novel=0
	p=len(test_user_movie[u])
	for j in test_user_movie[u]:
		if qi.has_key(j) and i!=j:
			novel=novel+1.0/float(p-1)*np.linalg.norm(qi[i]-qi[j])	
	return novel

def augMax(dict):
	max=0
	i=0
	for indi in dict:
		if dict[indi]>max:
			max=dict[indi]
			i=indi
	return i

def rTu(u,k):
	for i in test_user_movie[u]:
		if qi.has_key(i):
			test_user_movie[u][i]=novelty(u,i)
		else:
			test_user_movie[u][i]=0
	Tuk=set([])
	if len(test_user_movie[u])<k:
		kk=len(test_user_movie[u])
	else:
		kk=k
	for indi in range(kk):
		i=augMax(test_user_movie[u])
		del test_user_movie[u][i]
		Tuk.add(i)
	return Tuk
