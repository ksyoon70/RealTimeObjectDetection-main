import onnx_graphsurgeon as gs
import onnx
import numpy as np

print ("Patching the ONNX model.. ")

graph = gs.import_onnx(onnx.load("model.onnx"))
for inp in graph.inputs:
    inp.dtype = np.float32

onnx.save(gs.export_onnx(graph),"updated_model.onnx")

print ("Check ONNX model using checker function and see if it passes...")
model = onnx.load("updated_model.onnx")
onnx.checker.check_model(model)
print('The model is checked!') 