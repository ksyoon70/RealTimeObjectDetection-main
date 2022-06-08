# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 18:43:08 2022

@author: headway
"""
"""
import torch
import numpy as np
import onnxruntime as rt

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
def test():
    #model_pytorch = Net() # 네트워크 선언 및 가중치 로드 했다 치고..
    #x = torch.rand(b, c, h, w)
    
    #out_torch = model_pytorch(x)
    
    sess = rt.InferenceSession("model.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    
    print('inputname : {0} label_name : {1}'.format(input_name,label_name))
    
    #out_onnx = sess.run(None, {input_name: x})
    
    #np.testing.assert_allclose(to_numpy(out_torch), out_onnx[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    
test()
"""

import onnx 
from onnx import numpy_helper 
from onnx import helper
 # Load the ONNX model 
model = onnx.load("model.onnx") 
 # Print a human readable representation of the graph 
print(onnx.helper.printable_graph(model.graph)) 
# Check the model 
onnx.checker.check_model(model) 
print('The model is checked!')

