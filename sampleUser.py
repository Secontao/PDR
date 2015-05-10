user_Max_ID=6040
import random
l=random.sample(range(user_Max_ID),200)
print l

import pickle
output=open('user_sampling.pkl','w')
pickle.dump(l,output)
output.close()
