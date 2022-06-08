import tensorrt as trt

 

onnx_file_name = 'converted_int32_model.onnx'

tensorrt_file_name = 'bert.plan'

shape = [1,320,320,3]

fp_16_mode = True

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

 

builder = trt.Builder(TRT_LOGGER)

network = builder.create_network(EXPLICIT_BATCH)

parser = trt.OnnxParser(network, TRT_LOGGER)

#builder.max_workspace_size = (1 << 30)
config = builder.create_builder_config()
config.max_workspace_size = (1 << 30)

#builder.fp16_mode = fp16_mode
builder.max_batch_size = 8
builder.fp16_mode = True
builder.int8_mode = True


with open(onnx_file_name, 'rb') as model:

    if not parser.parse(model.read()):

        for error in range(parser.num_errors):

            print (parser.get_error(error))
    network.get_input(0).shape = shape


#engine = builder.build_cuda_engine(network)
engine = builder.build_engine(network, config)

buf = engine.serialize()

with open(tensorrt_file_name, 'wb') as f:

    f.write(buf)