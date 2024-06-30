#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:55:06 2020

@author: luqiqi
"""

import numpy as np

def res_stack(res_sigma,id_s0):
    res_sigma_reshape = np.reshape(res_sigma,[res_sigma.shape[0],res_sigma.shape[1],res_sigma.shape[2],10,10])
    
    res_stack = res_sigma_reshape[0,id_s0,0,:,:]

    for id_t2 in range(res_sigma.shape[2]):
        for id_noise_level in range(res_sigma.shape[0]):
            res_stack = np.vstack((res_stack,res_sigma_reshape[id_noise_level,id_s0,id_t2,:,:]))
            
    res_stacks = res_stack[10:(res_sigma.shape[0]+1)*10,:]
    
    for id_t2 in range(res_sigma.shape[2]):
        start = 10+res_sigma.shape[0]*10*id_t2
        
        end   = 10+res_sigma.shape[0]*10*id_t2 + res_sigma.shape[0]*10
        res_stacks = np.hstack((res_stacks,res_stack[start:end,:]))
        # print(start)
        # print(end)
    
    res_stacks = res_stacks[:,10:]
    return res_stacks