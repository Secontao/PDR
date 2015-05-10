import pickle
import numpy as np

input3=open('./pkl/test_user_movie.pkl')
test_user_movie=pickle.load(input3)


def rTu(qi,u):
	Tuk=set([])
	for i in test_user_movie[u]:
		if qi.has_key(i):
			Tuk.add(i)
	return Tuk
