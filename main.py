%pylab inline
from matplotlib.pyplot import style
style.use('https://raw.githubusercontent.com/JoseGuzman/minibrain/master/minibrain/paper.mplstyle')
import pandas as pd

data = pd.read_csv('dataset.csv', index_col=0)
data.head()
print(f'The dataset contains {data.shape[0]} observations')

import torch
import torch.nn as nn
from tqdm import tqdm # progress bar

torch.manual_seed(42); # set seed for reproducibility of results

GRE = torch.tensor( data = data.GRE.values, dtype = torch.float ) # x_1 
TOEFL = torch.tensor( data = data.TOEFL.values, dtype = torch.float) # x_2
SOP = torch.tensor( data = data.SOP.values, dtype = torch.float) # x_3
LOR = torch.tensor( data = data.LOR.values, dtype = torch.float) # x_4
CGPA = torch.tensor( data = data.CGPA.values, dtype = torch.float) # x_5
Research = torch.tensor( data = data.Research.values, dtype = torch.float) # x_6
UniversityRating = torch.tensor( data = data.UniversityRating.values, dtype = torch.float) # x_7

ChanceOfAdmit = torch.tensor( data = data.ChanceOfAdmit.values, dtype = torch.float ) # target

a = torch.randn(1, requires_grad = True)  # start with a random number from a normal distribution
b = torch.randn(1, requires_grad = True)
c = torch.randn(1, requires_grad = True)
d = torch.randn(1, requires_grad = True)
e = torch.randn(1, requires_grad = True)
f = torch.randn(1, requires_grad = True)
g = torch.randn(1, requires_grad = True)
h = torch.randn(1, requires_grad = True)

def mylnmodel(GRE: torch.tensor, TOEFL: torch.tensor, SOP: torch.tensor, LOR: torch.tensor, CGPA: torch.tensor, Research: torch.tensor, UniversityRating: torch.tensor):
    """
    computes f(x; a,b,c) = a + bx_1 + cx_2 + dx_3, 
    for independent variables x_1, x_2 and x_3.
    
    Arguments:
    tv (tensor) with the values of tv investment (x_1)
    radio (tensor) with the values of radio investment (x_2)
    news (tensor) with the newspaper investment (x_3).
    
    Note: coefficients a, b, c and d must be previoulsy 
    defined as tensors with requires_grad = True
    
    Returns a tensor with the backward() method
    """
    return a + b*GRE + c*TOEFL + d*SOP + e*LOR + f*CGPA + g*Research + h*UniversityRating

# generate the first prediction
predicted = mylnmodel(GRE, TOEFL, SOP, LOR, CGPA, Research, UniversityRating)
predicted.shape

# compare it with targets
ChanceOfAdmit.shape

plt.figure(figsize=(6,6))
plt.scatter(ChanceOfAdmit, predicted.detach(), c='k', s=4)
plt.xlabel('ChanceOfAdmit'), plt.ylabel('predicted');
x = y = range(100)
plt.plot(x,y, c='brown')
plt.xlim(0,100), plt.ylim(0,120);
plt.text(60,50, f'a     = {a.item():2.4f}', fontsize=10);
plt.text(60,45, f'GRE    = {b.item():2.4f}', fontsize=10);
plt.text(60,40, f'TOEFL = {c.item():2.4f}', fontsize=10);
plt.text(60,35, f'SOP    = {d.item():2.4f}', fontsize=10);
plt.text(60,30, f'LOR    = {e.item():2.4f}', fontsize=10);
plt.text(60,25, f'CGPA    = {f.item():2.4f}', fontsize=10);
plt.text(60,20, f'Research    = {g.item():2.4f}', fontsize=10);
plt.text(60,15, f'UniversityRating    = {h.item():2.4f}', fontsize=10);

def MSE(y_predicted:torch.Tensor, y_target:torch.Tensor):
    """
    Returns a single value tensor with 
    the mean of squared errors (SSE) between the predicted and target
    values:
    
    """
    error = y_predicted - y_target # element-wise substraction
    return torch.sum(error**2 ) / error.numel() # mean (sum/n)

predicted = mylnmodel(GRE, TOEFL, SOP, LOR, CGPA, Research, UniversityRating)
loss = MSE(y_predicted = predicted, y_target=ChanceOfAdmit)
print(loss)

# initial values for the coefficients is random, gradients are not calculated
print(f'a = {float(a.item()):+2.4f}, df(a)/da = {a.grad}') # 0.3367
print(f'b = {float(b.item()):+2.4f}, df(b)/da = {a.grad}') # 0.1288
print(f'c = {float(c.item()):+2.4f}, df(c)/dc = {c.grad}') # 0.2345
print(f'd = {float(d.item()):+2.4f}, df(d)/dd = {d.grad}') # 0.2303
print(f'e = {float(e.item()):+2.4f}, df(e)/de = {e.grad}') # 0.2303
print(f'f = {float(f.item()):+2.4f}, df(f)/df = {f.grad}') # 0.2303
print(f'g = {float(g.item()):+2.4f}, df(g)/dg = {g.grad}') # 0.2303
print(f'h = {float(h.item()):+2.4f}, df(h)/dh = {h.grad}') # 0.2303

loss.backward()

# initial values for the coefficients is random, gradients are now calculated
print(f'a = {float(a.item()):+2.4f}, df(a)/da = {a.grad}') # 0.3367
print(f'b = {float(b.item()):+2.4f}, df(b)/da = {a.grad}') # 0.1288
print(f'c = {float(c.item()):+2.4f}, df(c)/dc = {c.grad}') # 0.2345
print(f'd = {float(d.item()):+2.4f}, df(d)/dd = {d.grad}') # 0.2303
print(f'e = {float(e.item()):+2.4f}, df(e)/de = {e.grad}') # 0.2303
print(f'f = {float(f.item()):+2.4f}, df(f)/df = {f.grad}') # 0.2303
print(f'g = {float(g.item()):+2.4f}, df(g)/dg = {g.grad}') # 0.2303
print(f'h = {float(h.item()):+2.4f}, df(h)/dh = {h.grad}') # 0.2303

## Use gradiendt descent
myMSE = list()
for i in tqdm(range(5_000)):
    a.grad.zero_()
    b.grad.zero_()
    c.grad.zero_()
    d.grad.zero_()
    e.grad.zero_()
    f.grad.zero_()
    g.grad.zero_()
    h.grad.zero_()
    
    predicted = mylnmodel(GRE, TOEFL, SOP, LOR, CGPA, Research, UniversityRating)
    loss = MSE(y_predicted = predicted, y_target = ChanceOfAdmit) # calculate MSE
    
    loss.backward() # compute gradients
    myMSE.append(loss.item()) # append loss
    with torch.no_grad():
        a -= a.grad * 1e-6
        b -= b.grad * 1e-6
        c -= c.grad * 1e-6
        d -= d.grad * 1e-6
        e -= e.grad * 1e-6
        f -= f.grad * 1e-6
        g -= g.grad * 1e-6
        h -= h.grad * 1e-6
        
plt.plot(myMSE);
plt.xlabel('Epoch (#)'), plt.ylabel('Mean squared Errors')