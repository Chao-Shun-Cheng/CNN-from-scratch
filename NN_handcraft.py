#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
def shuffle(x,y):
    x = x.T
    y = y.T
    combine = np.c_[x.reshape(len(x), -1), y.reshape(len(y), -1)]
    np.random.shuffle(combine)
    return combine[:, :x.size//len(x)].reshape(x.shape).T,combine[:, x.size//len(x):].reshape(y.shape).T
class One_layer:    
    def __init__(self, in_dim, out_dim, act,lr):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = np.random.randn(out_dim,in_dim+1)/10
        #self.weight = np.random.rand(out_dim,in_dim+1)/10
        
        act_list = {"relu":self.sigmoid, "tanh":self.tanh,"relu":self.relu,"sigmoid":self.sigmoid,
                    "softmax":self.softmax,"linear":self.linear}
        dif_list = {"relu":self.dif_sigmoid, "tanh":self.dif_tanh,"relu":self.dif_relu,"sigmoid":self.dif_sigmoid,
                    "softmax":self.dif_softmax,"linear":self.dif_linear}
        self.act = act_list[act]
        self.dif_act = dif_list[act]
        self.lr = lr
        #self.batch_size = batch_size
        self.input = np.array([]) #np.zeros((batch_size,in_dim))
        self.local_grad = np.array([]) #np.zeros((batch_size,out_dim))    
        self.value = np.array([])
        
    def forward(self,data):    
        data = np.concatenate((np.ones(data.shape[1]).reshape(1,-1),data), axis=0)         
        self.input = data
        self.value = self.weight.dot(data)
        return self.act(self.value) 
    
    def output(self,data): 
        data = np.concatenate((np.ones(data.shape[1]).reshape(1,-1),data), axis=0) 
        return self.act(self.weight.dot(data))    
    
    def backward(self,err):
        self.local_grad = self.dif_act(self.value,err)
        return self.weight.T.dot(self.local_grad)[1:,:]    
    
    def weight_renew(self,batch_size):       
        delta_w = self.lr*(self.local_grad).dot(self.input.T)#/batch_size#
        self.weight = self.weight - delta_w
        
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def tanh(self,z):
        return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    def relu(self,z):
        return np.maximum(0,z)  
    
    def linear(self,z):
        return z 
    
    def dif_linear(self,z,err):
        return err
    
    def softmax(self,x):
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / np.sum(exps,axis=0)    
    
    def dif_sigmoid(self,z,err):
        temp=self.sigmoid(z)        
        return (temp*(1-temp))*err    
    def dif_tanh(self,z,err):
        return (1-self.tanh(self,z)**2)*err    
    def dif_relu(self,z,err):
        return np.maximum(np.sign(z),0)*err     
    def dif_softmax(self,z,err):
        local_grad = []
        for i,ee in enumerate(z.T):
            ee = ee.T.reshape(1,-1)
            tt = self.softmax(ee.T)
            local_grad.append((np.eye(ee.size)*tt-tt.dot(tt.T)).dot(err[:,i]))
        return np.array(local_grad).T#(np.eye(self.out_dim)*S - S.dot(S.T)).dot(err)
    def fetch_w(self):
        return self.weight
    
class RBF_layer:    
    def __init__(self, in_dim, out_dim, cent_lr, wid_lr,data,act="gaussian"):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.centers = self.plus_plus(data, out_dim)
        self.var_sqr_inv = np.ones((out_dim,in_dim))/10
        
        act_list = {"gaussian":self.gaussian, "inv_quadratic":self.quadratic}
        act_diff_list = {"gaussian":self.diff_gaussian, "inv_quadratic":self.diff_quadratic}
        
        self.act = act_list[act]
        self.act_diff = act_diff_list[act]
        
        self.cent_lr = cent_lr
        self.wid_lr = wid_lr
        self.output_norm1 = []
        self.out = []
        self.err_diff = []
    def gaussian(self,data):
        return np.exp(-data)
    def quadratic(self,data):
        return 1/(1+data)**0.5
    
    def forward(self,data): 
        o1 = []
        for i,cc in enumerate(self.centers):
            o1.append((data.T-cc))            
        self.output_norm1 = np.array(o1) 
        
        ot = []          
        for i,dc in enumerate(np.array(o1)):
            ot.append((dc**2).dot(self.var_sqr_inv[i].reshape(-1,1))[:,0])        
        temp =  self.act(np.array(ot))
        self.out = temp
        return temp    
    def output(self,data): 
        o1 = []
        for i,cc in enumerate(self.centers):
            o1.append((data.T-cc))             
        ot = []           
        for i,dc in enumerate(np.array(o1)):
            ot.append((dc**2).dot(self.var_sqr_inv[i].reshape(-1,1))[:,0])  
        temp =  self.act(np.array(ot))
        #self.out = temp
        return temp  
    
    def backward(self,err):
        self.err_diff = err
        return 0 
    
    def weight_renew(self,batch_size):   
        self.act_diff()
    
    def diff_quadratic(self):
        delta_c = []
        for i,dc in enumerate(self.output_norm1):
            delta_c.append(-(2*dc.T*self.var_sqr_inv[i].reshape(-1,1)).dot((self.err_diff[i]*-0.5*self.out[i]**3).reshape(-1,1))[:,0])
        delta_c = np.array(delta_c)
        
        delta_wid = []
        for i,dc in enumerate(self.output_norm1):
            delta_wid.append((self.err_diff[i]*-0.5*self.out[i]**3).dot(dc**2))
        delta_wid = np.array(delta_wid)
        
        self.centers -= self.cent_lr*delta_c
        self.var_sqr_inv = np.maximum(0,self.var_sqr_inv- self.wid_lr*delta_wid) 
        
    def diff_gaussian(self):
        delta_c = []
        for i,dc in enumerate(self.output_norm1):
            delta_c.append((2*dc.T*self.var_sqr_inv[i].reshape(-1,1)).dot((self.err_diff[i]*self.out[i]).reshape(-1,1))[:,0])
        delta_c = np.array(delta_c)
        
        delta_wid = []
        for i,dc in enumerate(self.output_norm1):
            delta_wid.append(-(self.err_diff[i]*self.out[i]).dot(dc**2))
        delta_wid = np.array(delta_wid)
        
        self.centers -= self.cent_lr*delta_c
        self.var_sqr_inv = np.maximum(0,self.var_sqr_inv- self.wid_lr*delta_wid) 
        
    def plus_plus(self,ds, k):
        centroids = [ds[0]]
        for _ in range(1, k):
            dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in ds])
            probs = dist_sq/dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break
            centroids.append(ds[i])
        return np.array(centroids)    
    
class NN_model:
    def __init__(self,layer_list,batch_size,loss):
        cost_list = {"Binary":self.CE_twoclass, "Multi_class":self.CE,
                     "Regression_mse":self.MSE,"Regression_mae":self.MAE}     
        cost_dif_list = {"Binary":self.dif_ce_twoclass, "Multi_class":self.dif_ce,
                         "Regression_mse":self.dif_mse,"Regression_mae":self.dif_mae}
        self.layer_list = layer_list
        self.batch_size = batch_size
        self.loss=cost_list[loss]
        self.dif_loss=cost_dif_list[loss]
        
        acc_type_list = {"Binary":self.acc_binary, "Multi_class":self.acc_multiclass
                         ,"Regression_mae":self.acc_regression,"Regression_mse":self.acc_regression}
        self.loss_func = acc_type_list[loss]
        
    def output(self,x):
        temp = x
        for ll in self.layer_list:
            temp = ll.output(temp)
        return temp
    
    def forward(self,x):
        temp = x
        for ll in self.layer_list:
            temp = ll.forward(temp)            
        return temp
    
    def back_propagation(self,x):
        temp = x
        for ll in self.layer_list[::-1]:
            temp = ll.backward(temp)
        return temp
    
    def loss_dif(self,y,y_hat):  
        return self.dif_loss(y,y_hat)    
    
    def MSE(self,y,y_hat,scale):
        err_avg = np.sum(((y_hat-y)*scale[1])**2,axis=0)
        return err_avg,sum(err_avg)/y_hat.shape[1]
    
    def MAE(self,y,y_hat,scale):
        err_avg = np.sum(np.abs((y_hat-y)*scale[1]),axis=0)
        return err_avg,sum(err_avg)/y_hat.shape[1]
    
    def CE_twoclass(self,p,y):
        err = -(y*np.log(p)+(1-y)*np.log(1-p))
        each_batch_avg_err = np.sum(err,axis=1)/err.shape[1]
        total_avg_err = sum(each_batch_avg_err)
        return err,total_avg_err
    
    def CE(self,p,y):
        err = -np.log(np.sum(p*y,axis=0))
        return err,sum(err)/p.shape[1]
    
    def dif_ce_twoclass(self,p,y):            
        #return (np.sum(-(np.divide(y, p) - np.divide(1 - y, 1 - p)),axis=1)/y.shape[1]).reshape(-1,1) 
        return (-(np.divide(y, p) - np.divide(1 - y, 1 - p)))
    
    def dif_ce(self,p,y):
        return -np.divide(1,p)*y 
    
    def dif_mae(self,y,y_hat):
        #return -(np.sum(y_hat-y,axis=1)/y.shape[1]).reshape(-1,1)
        return -np.sign((y_hat-y))
    
    def dif_mse(self,y,y_hat):
        #return -(np.sum(y_hat-y,axis=1)/y.shape[1]).reshape(-1,1)
        return -(y_hat-y)
    
    def update_weight(self):
        for ll in self.layer_list:
            ll.weight_renew(self.batch_size)
#Kenny         return self.layer_list[len(self.layer_list)-1].weight
    def shuffle(self,x,y):
        x = x.T
        y = y.T
        combine = np.c_[x.reshape(len(x), -1), y.reshape(len(y), -1)]
        np.random.shuffle(combine)
        return combine[:, :x.size//len(x)].reshape(x.shape).T,combine[:, x.size//len(x):].reshape(y.shape).T
    
    # -------------------------------------- Acc --------------------------------------------------#
    def acc_binary(self,ee,epochs,x_train,x_valid,y_train,y_valid,scale=[]):
        y_va = self.output(x_valid)
        vl = self.loss(y_va,y_valid)[1]
        y_tr = self.output(x_train)
        tl = self.loss(y_tr,y_train)[1]        
        ta = self.bb_acc(y_tr,y_train)
        va = self.bb_acc(y_va,y_valid)        
        print("Epoch",ee,"/",epochs,"- loss: %.4f" %tl,
            "- accuracy: %.4f" %ta,
            "- val_loss: %.4f" %vl,
            "- val_accuracy: %.4f" %va)            
        return [tl,vl,ta,va]
    
        out =(yhat> 0.5).astype(int)
        return np.sum(np.equal(y.reshape(-1,1),out.reshape(-1,1)))/y.size
    
    def bb_acc(self,yhat,y):
        out =(yhat> 0.5).astype(int)
        return np.sum(np.equal(y.reshape(-1,1),out.reshape(-1,1)))/y.size
    
    def acc_multiclass(self,ee,epochs,x_train,x_valid,y_train,y_valid,scale=[]):
        y_va = self.output(x_valid)
        vl = self.loss(y_va,y_valid)[1]
        y_tr = self.output(x_train)
        tl = self.loss(y_tr,y_train)[1]         
        ta = self.mc_acc(y_tr,y_train)
        va = self.mc_acc(y_va,y_valid)        
        print("Epoch",ee,"/",epochs,"- loss: %.4f" %tl,
            "- accuracy: %.4f" %ta,
            "- val_loss: %.4f" %vl,
            "- val_accuracy: %.4f" %va)            
        return [tl,vl,ta,va]
    
    def mc_acc(self,yhat,y):
        y_hat = np.argmax(yhat,axis=0)
        yy = np.argmax(y,axis=0)
        return np.sum(np.equal(y_hat,yy))/len(yy)
    
    def acc_regression(self,ee,epochs,x_train,x_valid,y_train,y_valid,scale):
        y_va = self.output(x_valid)
        vl = self.loss(y_va,y_valid,scale)[1]
        y_tr = self.output(x_train)
        tl = self.loss(y_tr,y_train,scale)[1] 
        print("Epoch",ee,"/",epochs,"- loss: %.4f" %tl,
              "- val_loss: %.2f " %vl) 
        return [tl,vl]
    
    def training(self,x_train,x_valid,y_train,y_valid,epochs,scale=[]):        
        batches = int(x_train.shape[1]/self.batch_size)
        max_index = x_train.shape[1]    
        history = []
        w_list = []
        for ee in range(epochs):
            x_train,y_train = self.shuffle(x_train,y_train) ## 每次training data 的順序打亂
#             print("start")
            for batch in range(batches):
#                 print("start")
                st = batch*self.batch_size
                end = min((st+self.batch_size),max_index)                  
                out = self.forward(x_train[:,st:end])
                err = self.loss_dif(out,y_train[:,st:end])
                self.back_propagation(err)
#                 w_list.append(self.update_weight())
                self.update_weight()
#                 print("end")
#             print("ok")
            history.append(self.loss_func(ee,epochs,x_train,x_valid,y_train,y_valid,scale))
        return np.array(history)

