import time
t1=time.clock()

maxMovieId=3952
maxUserID=6040
import pickle
import numpy as np

input1=open('pu.pkl')
input2=open('qi.pkl')
input3=open('user_movie.pkl')

pu=pickle.load(input1)
qi=pickle.load(input2)
user_movie=pickle.load(input3)

print 'time1:%f' % (time.clock()-t1)

def d(i,j,u,Beta):
	return np.dot(pu[u].reshape(1,-1),qi[i])[0][0]+np.dot(pu[u].reshape(1,-1),qi[j])[0][0]+Beta*np.linalg.norm(qi[i]-qi[j])

def mmd(u,k,Beta):
	s=set([])
	I=set(xrange(1,maxMovieId+1))
	Omega=set([])
	for index in user_movie[u]:
		Omega.add(index)
	Theta=I-Omega
	dist=np.zeros((maxMovieId,maxMovieId))

	print 'time2:%f' % (time.clock()-t1)

	for indi in Theta:
		for indj in Theta:
			if indi!=indj and qi.has_key(indi) and qi.has_key(indj):
				dist[indi-1][indj-1]=d(indi,indj,u,Beta)
	
	print 'time3:%f' % (time.clock()-t1)

	print len(Theta)
	for index in xrange(k//2):
		max=0
		i=0
		j=0
		for indi in Theta:
			for indj in Theta:	
				if indi!=indj and qi.has_key(indi) and qi.has_key(indj) and dist[indi-1][indj-1]>max:
					max=dist[indi-1][indj-1]	
					i=indi
					j=indj
		s.add(i)
		s.add(j)
		Theta.remove(i)	
		Theta.remove(j)
	if k%2!=0:
		s.add(Theta.pop())
	print s

	print 'time4:%f' % (time.clock()-t1)

#for u in xrange(1,maxUserId+1):
mmd(1,10,0.15)
