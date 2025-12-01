import onnx

model = onnx.load("../../models/Llama-3.2-3B-Instruct-ONNX/cuda/cuda-fp16/model.onnx")
for i, inp in enumerate(model.graph.input):
    print(i, inp.name)
for i, out in enumerate(model.graph.output):
    print(i, out.name)