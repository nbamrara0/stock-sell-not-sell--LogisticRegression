from sklearn.linear_model import LogisticRegression
import numpy as np


#data
# stock price
x=np.array([[1],[5],[10],[15],[20],[25],[30],[35],[40]])
# stock sell/not sell
y=np.array([0,0,0,1,1,1,1,1,1])

# model
model= LogisticRegression()
model.fit(x,y)


# prediction
prediction=model.predict([[17]])
if prediction==1:
    print ("sell")
else:print ("not sell")


#probility
prob = model.predict_proba([[17]])
print("probability:", prob)
