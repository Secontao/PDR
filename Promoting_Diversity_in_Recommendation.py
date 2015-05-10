'''

Reference paper:
	"Promoting Diversity in Recommendation" 
		Sha

	the PDR_alpha_beta means alpha>0,beta>0
Created by Sect(2014)

'''
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
	sigma_E^2:
	sigma_U^2:
Output:
	the top-k Recommendation set S 
'''
##################################################

class PDR():
	def __init__(self,U,V,user_movie,movie_user,u,sigmaE2,sigmaU2,k,alpha,beta,lamb):
		self.U=U
		self.V=V
		self.user_movie=user_movie
		self.movie_user=movie_user
		self.u=u
		self.sigmaE2=sigmaE2
		self.sigmaU2=sigmaU2
		self.k=k
		self.alpha=alpha
		self.beta=beta
		self.lamb=lamb
		return
	
	def maxInS(V,S,j):
		max=0
		for indl in S:
			med=np.dot(V[indl,:],V[j,:])
			if med>max:
				max=med
		return max
		
	def Max(a,b):
		if a>b:
			return a
		else:
			return b
	
	def W(rating,j):
		return math.e**(rating[j]-rating_MAX)

	def SIG_SS(SIG,S,sigmaE2):
		x=list(S)
		x=np.array(x)-1
		x=list(x)
		SIG_xx=(SIG[x,:][:,x]+sigmaE2*np.eye(len(S)))
		SIG_xx=np.mat(SIG_xx)
		SIG_xx=SIG_xx.I
		SIG_xx=np.array(SIG_xx)
		return SIG_xx

	def TOP_k(self):
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
				p_D={}
				p_R={}
				for w in Theta:
					if movie_user.has_key(w) and len(movie_user[w])>10:	
						#Rating	
						o=list(Omega)
						o=np.array(o)-1
						o=list(o)	
						SIG_wO=SIG[w-1,o]
						p_R[w]=np.dot(np.dot(SIG_wO,D),rOmega)

						#Rating refined by alpha
						if alpha>0:
							for j in Omega:
								p_R[w]=p_R[w]+alpha*W(user_movie[u],j)*Max(0,np.dot(V[w,:],V[j,:])-maxInS(V,S,j))

						#Diversity by beta
						if beta>0:
							for j in S:
								p_D[w]+=beta*np.linalg.norm(V[w-1,:]-V[j-1,:],2)

						#Diversity by Entropy_Regularizer
						if lamb>0:
							a=list(A)
							a=np.array(a)-1
							a=list(a)	
							SIG_wA=SIG[w-1,a]
							p_D[w]=math.log(1*2*math.pi*math.e*(sigmaE2+np.dot(np.dot(SIG_wA,C),SIG_wA.transpose())))

				max=0
				w=0

				for j in Theta:
					if movie_user.has_key(j)and len(movie_user[j])>10:
						med=p_R[j]+p_D[j]
						if med>max:
							max=med
							w=j
				S.add(w)
				A.add(w)
				Theta.remove(w)
		return S
