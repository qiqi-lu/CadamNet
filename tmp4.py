# import tensorflow as tf
import numpy as np
# import helper
# import os
import matplotlib.pyplot as plt
# import config

meanR2=np.linspace(10.0,1500,200)

weiw = 1.0-0.5*np.exp(-1.0*meanR2/200.0)
weip = 0.5+0.5*np.exp(-1.0*meanR2/800.0)

weiw = 4.0*np.exp(-meanR2/200.0)+1.0
weiw = np.exp(meanR2/200.0)
weiw = 1.5*np.exp(-meanR2/300.0)+0.5

ref  = 1.0/(1.0+np.exp((meanR2-500.0)/100.0))
ref  = 1.5*np.exp(-meanR2/300.0)+0.5

plt.figure()
plt.plot(meanR2,weiw,label='w')
plt.plot(meanR2,weip,label='p')
plt.plot(meanR2,ref ,label='ref')

plt.plot(meanR2,np.zeros(meanR2.shape),'--k')
plt.plot([1000,1000],[0,1],'--k')
plt.plot([0,1500],[0.5,0.5],'--k')
plt.legend()
plt.savefig('figure/wei.png')