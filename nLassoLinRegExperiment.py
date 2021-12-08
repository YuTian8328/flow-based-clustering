# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import networkx as nx 

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(n, 1,bias=False) # 2 in and 1 out
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

clustersizes = [5, 5, 5]
nrclusters = len(clustersizes)
probs = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.80]]

g = nx.stochastic_block_model(clustersizes, probs, seed=0)
nrnodes = len(g.nodes(data=False))
#print(nrnodes)

ndecntr=0
attrlabel=dict()

n = 2
m = 5

# generate ground truth weight vectors. one weight vector for each cluster 
# the weight vectors are then used to generate labels from features 

weights = np.array(np.random.randn(nrclusters,n), dtype=np.float32) ; 

for clusteridx in range(nrclusters): 
    wcurr = weights[clusteridx,:] 
    for dmy in range(clustersizes[clusteridx]):
        X = np.array(np.random.randn(m,n), dtype=np.float32) 
        y = X.dot(wcurr).reshape(-1,1)
        x_data = Variable(torch.from_numpy(X))
        y_data = Variable(torch.from_numpy(y))
        model = Model()
        criterion = torch.nn.MSELoss(size_average=True)
        optimizer = torch.optim.RMSprop(model.parameters())
        winit = np.array(np.random.randn(1,n), dtype=np.float32) ;  #np.zeros(wcurr.shape)
        attrlabel.update({ndecntr:{"weight":winit,"cluster":clusteridx,"features":x_data,"label":y_data,"model":model,"optimizer":optimizer,"criterion":criterion}}) 
        ndecntr=ndecntr+1
        
attrdegree=dict()

for nodedmy in g.nodes(data=False):
    tmp = torch.from_numpy(np.array(g.degree(nodedmy)/2))
    attrdegree.update({nodedmy:{"degree":tmp}}) 
        
nx.set_node_attributes(g, attrlabel)
nx.set_node_attributes(g, attrdegree)

for edgevar in g.edges():
    g.edges[edgevar]["weight"] =  np.zeros((1,n)) #np.array(np.random.randn(1,n), dtype=np.float32) ; 
    

labels = nx.get_node_attributes(g,"cluster")

#print(g.nodes[0])


### implement algorithm 1 of papger []]

lambda_nlasso=1/10
    
for iter_algo in range(20):
  #  graphsigold = np.fromiter(nx.get_node_attributes(g,'weight').values(),dtype=float, count=nrnodes)
   oldnodeweight=nx.get_node_attributes(g,"weight")  # read in current node weight vectors
   for nodevar in g.nodes(data=False): 
     #   print("node =",nodevar)
        oldweight = oldnodeweight[nodevar]
        dmy=np.zeros(oldweight.shape)
        # iterate over all neighbors of node "nodevar"
        for node_j in g[nodevar]: 
        #    print(nodevar,node_j)
            edgeweighttmp =  g.edges[nodevar,node_j]["weight"]
            if node_j > nodevar:
             dmy = dmy-edgeweighttmp#/(fac1*fac2)
            else: 
             dmy = dmy+edgeweighttmp
        dmy= dmy/g.degree(nodevar); 
        regularizerterm = oldweight - dmy 
        
        optimizer = g.nodes[nodevar]["optimizer"]
        model = g.nodes[nodevar]["model"]
        text = "iter = %d\n , test = %f\n" % (iter_algo, np.linalg.norm(model.linear.weight.data-oldweight))
        print(text)
        model.linear.weight.data=torch.from_numpy( np.array(oldweight, dtype=np.float32)) #oldweight)
       
        y_data = g.nodes[nodevar]["label"]
        x_data = g.nodes[nodevar]["features"]
        criterion = g.nodes[nodevar]["criterion"]
        taufac = g.nodes[nodevar]["degree"]
        # Zero gradients, perform a backward pass, and update the weights.
        for iterinner in range(40):
            def closure():
                optimizer.zero_grad()
                y_pred = model(x_data) 
    # Compute and print loss
                loss1 = criterion(y_pred, y_data)
                loss= loss1+1*torch.mean((model.linear.weight-torch.from_numpy(regularizerterm))**2) #+ 10000*torch.mean((model.linear.bias+0.5)**2)#model.linear.weight.norm(2)
          #   print(iter_algo, loss1.data.numpy())
             #   print("iter ",iterinner,loss.data.numpy())
                loss.backward()
                return loss
            optimizer.step(closure)
       
        wtmp =  list(model.parameters())
     #   print("wtmp:",wtmp)
        g.nodes[nodevar]["weight"] =model.linear.weight.data.numpy() #regularizerterm#wtmp[0].data.numpy()
        
  # print(oldnodeweight)
 #  print(nx.get_node_attributes(g,"weight")  )
  # print(nx.get_edge_attributes(g,"weight")  )
        
   for edgevar in g.edges(data=False): 
        tmpweight =  g.edges[edgevar]["weight"]
        tailnode=np.min([edgevar[0],edgevar[1]])
        headnode=np.max([edgevar[0],edgevar[1]])
      # print("tailnode :%d"%tailnode,"headnode: %d"%headnode)
        tmpnodew1 = 2*g.nodes[headnode]["weight"]-oldnodeweight[headnode]
        tmpnodew2 = 2*g.nodes[tailnode]["weight"]-oldnodeweight[tailnode]
        tmp =  tmpweight+(1/2.1)*(tmpnodew1-tmpnodew2)
      #  print(tmp)
        g.edges[edgevar]["weight"] = np.clip(tmp,-lambda_nlasso,lambda_nlasso)
    
#   print(nx.get_edge_attributes(g,"weight")  )
        
    
   print("completed iteration nr. ",iter_algo)
#nx.draw(g,labels=labels)

currweights = nx.get_node_attributes(g,"weight")
for nodeidx in range(nrnodes):
    print(currweights[nodeidx])


## LogSoftmax + ClassNLL Loss
#
#
#def my_loss(output, target):
#   # loss_fn =  torch.nn.BCELoss(size_average=True)
#    err = torch.nn.MSELoss(output, target)
#    loss = err+0*torch.mean((output - target)**2)
#    return loss
#
#N = 100
#D = 1
#
#X = np.random.randn(N,D)*2
#
#N_half = int(np.ceil(N/2))
#
## center the first N/2 points at (-2,-2)
##X[:N_half,:] = X[:N_half,:] - 2*np.ones((N_half,D))
#
## center the last N/2 points at (2, 2)
##X[N_half:,:] = X[N_half:,:] + 2*np.ones((N_half,D))
#
## labels: first N/2 are 0, last N/2 are 1
##T = np.array([0]*(N_half) + [1]*(N_half)).reshape(100,1)
#T = X[:,0]
#
##x_data = Variable(torch.Tensor(X))
##y_data = Variable(torch.Tensor(T))
#
#
#
#
#
## create dummy data for training
#X = np.array(np.random.randn(m,n), dtype=np.float32) 
#w = np.zeros((n,1), dtype=np.float32)
#w[1]=3
#w[0]=-0.77
#y = X.dot(w)
#
#x_values = [i for i in range(11)]
#x_train = np.array(x_values, dtype=np.float32)
#x_train = x_train.reshape(-1, 1)
#
#y_values = [2*i + 1 for i in x_values]
#y_train = np.array(y_values, dtype=np.float32)
#y_train = y_train.reshape(-1, 1)
#
#
#
#
#    
## Our model    
model = Model()
#print("model.linear.weight=",model.linear.weight)
#
#refweight = torch.zeros(model.linear.weight.shape)
#refweight[0]=0.3
#print(model.linear.weight.shape)
#torch.vstack
criterion = torch.nn.MSELoss(size_average=True)
optimizer = torch.optim.LBFGS(model.parameters())

nodeattributes =nx.get_node_attributes(g,"features") 
nodeattributeslabel =nx.get_node_attributes(g,"label") 

#print(nodeattributes[0])

xall= torch.empty(0, n)
yall= torch.empty(0, 1)

for nodevar in g.nodes(data=False): 
    xall = torch.vstack((xall,nodeattributes[nodevar]))
    yall = torch.vstack((yall,nodeattributeslabel[nodevar]))
    
#print(xall)


 
#x_data = 

#
#
#x_data = Variable(torch.from_numpy(X))
#y_data = Variable(torch.from_numpy(y))
#
#
## Training loop
for epoch in range(10):
#    # print(type(x_data))
   def closure():
        optimizer.zero_grad()
        y_pred = model(xall) 
        loss1 = criterion(y_pred, yall)
        loss= (1/1000)*loss1#+10000*torch.mean((model.linear.weight-refweight)**2) + 10000*torch.mean((model.linear.bias+0.5)**2)#model.linear.weight.norm(2)
        loss.backward()
        return loss
#        
#    # Zero gradients, perform a backward pass, and update the weights.
#    
   optimizer.step(closure)
#
#
w = list(model.parameters())
print("weights :",weights)

#w0 = w[0].data.numpy()
#w1 = w[1].data.numpy()
#print(type(model.linear.weight))
#
#import matplotlib.pyplot as plt
#
print("Final gradient descend:", w)