#
#
#
#

import numpy as np
import time
import pickle
import cPickle
import gzip

maxMovieId=3952
maxUserID=6040

with gzip.open('features', 'rb') as f:
	_user_features = cPickle.load(f)
	_item_features = cPickle.load(f)
	train_user_item=cPickle.load(f)
	train_item_user=cPickle.load(f)
	validation_user_item=cPickle.load(f)
	PMF_lam=cPickle.load(f)
	num_user, num_feature_u = _user_features.shape
	num_item, num_feature_i = _item_features.shape
	print "num_user:",num_user
	print "num_item:",num_item
	print "PMF_lam:",PMF_lam
	print "num_feature:",num_feature_u 
	
	if num_feature_i != num_feature_u:
		raise DimensionError()
	_num_feature = num_feature_i

def diversity(S):
	div=0;
	for i in S:
		for j in S:
			if i!=j:
				div=div+1-np.dot(_item_features[int(i)-1,:],_item_features[int(j)-1,:])/np.linalg.norm(_item_features[int(i)-1,:],2)/np.linalg.norm(_item_features[int(j)-1,:],2)
	return 2*div/float(len(S))/float(len(S)-1)

def rTu(validation_user_movie,movie_user,u):
	Tu_k=set([])
	for i in validation_user_movie[u]:
		if movie_user.has_key(i):
			Tu_k.add(i)
	return Tu_k

import PDR_beta
import PDR_alpha_beta
import PDR_ER

sigma_e_2=1.0
sigma_u_2=2.0*float(sigma_e_2)/PMF_lam
k=30#the number of top-k recommendation
print "k:",k

#lamb=1000#for algrithomER
#print 'lamb:',lamb
#alpha=5
#print 'alpha:',alpha
#beta=0
#print 'beta:',beta

input=open('./pkl/user_sampling.pkl')
sample=pickle.load(input)

beta_list=[0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,1,5,10]
pre_beta_list=[]
div_beta_list=[]
for beta in beta_list:
	count=0
	number_sample=0
	sumDiv=0
	print 'beta:',beta
	for u in sample:
		if len(train_user_item[u])>30:
			t1=time.clock()
			Su=PDR_beta.Pre_Div(_user_features,_item_features,train_user_item,train_item_user,u,sigma_e_2,sigma_u_2,k,beta)
			#Su=PDR_alpha_beta.Pre_Div(_user_features,_item_features,train_user_item,train_item_user,u,sigma_e_2,sigma_u_2,k,alpha,beta)
			#Su=PDR_ER.ER(_user_features,_item_features,train_user_item,train_item_user,u,sigma_e_2,sigma_u_2,k,lamb)
			print 'Rec for user:',u,' ',Su
			print 'Diversity for this Rec:',diversity(Su)
			sumDiv=sumDiv+diversity(Su)
			#Tuk=rTuk.rTu(u,30)	
			Tu_k=rTu(validation_user_item, train_item_user, u)	
			print Tu_k
			count=count+len(Su & Tu_k)
			number_sample=number_sample+1
			print 'Average Diversity:',sumDiv/number_sample
			print 'Average Precision:',float(count)/float(k)/float(number_sample)
			print 'a user time:',time.clock()-t1
			if number_sample==10:
				break
	div_beta_list.append(sumDiv/number_sample)
	pre_beta_list.append(float(count)/float(k)/float(number_sample))
print pre_beta_list
print div_beta_list
	

