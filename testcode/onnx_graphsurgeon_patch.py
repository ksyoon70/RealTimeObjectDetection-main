import onnx
import onnx_graphsurgeon as gs
import numpy as np


graph = gs.import_onnx(onnx.load('converted_int32_model.onnx'))
nodes = graph.nodes
tensors = graph.tensors()

# set input_tensor shape & dtype
input_tensor = tensors['input_tensor']
input_tensor.dtype = np.float32
input_tensor.shape = [1, 320, 320, 3]

# # resize mode
# # 전처리 Loop 노드 내부에 서브 그래프가 존재함. - node.attrs['body']로 접근 
# preprocessing_node = nodes[2]
# resize_node = [node for node in preprocessing_node.attrs['body'].nodes if node.op == 'Resize'][0]
# resize_node.attrs['coordinate_transformation_mode'] = 'half_pixel'

# replace preprocessing node
# efficientNet 전처리 과정 구현
scale = gs.Constant(name='scale', values=np.array([1./255.], np.float32).reshape(1,))
input_scaled = gs.Variable(name='input_scaled', dtype=np.float32)
node_scale = gs.Node(op='Mul', inputs=[input_tensor, scale], outputs=[input_scaled])
nodes.append(node_scale)

ch_offset = gs.Constant(name='ch_offset', values=np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3))
input_ch_shifted = gs.Variable(name='input_ch_shifted', dtype=np.float32)
node_ch_shift = gs.Node(op='Sub', inputs=[input_scaled, ch_offset], outputs=[input_ch_shifted])
nodes.append(node_ch_shift)

ch_scale = gs.Constant(name='ch_scale', values=(1./np.array([0.229, 0.224, 0.225], np.float32)).reshape(1, 1, 3))
input_ch_scaled = gs.Variable(name='input_ch_scaled', dtype=np.float32)
node_ch_scale = gs.Node(op='Mul', inputs=[input_ch_shifted, ch_scale], outputs=[input_ch_scaled])
nodes.append(node_ch_scale)

# onnx의 Conv 노드의 입력은 NCHW 포맷이므로 이미지를 transpose한다.
input_transposed = gs.Variable(name='input_transposed', dtype=np.float32)
node_transpose = gs.Node(
    op='Transpose',
    attrs={'perm': [0, 3, 1, 2]},
    inputs=[input_ch_scaled],
    outputs=[input_transposed],
)
nodes.append(node_transpose)

# Conv 노드의 입력 중 Loop 노드로부터의 입력을 새로운 전처리 노드의 출력으로 대체한다.
conv_node = [n for n in nodes if n.name == 'StatefulPartitionedCall/EfficientDet-D0/model/stem_conv2d/Conv2D'][0]
conv_node.i(0).outputs.clear()
conv_node.inputs[0] = input_transposed

# raw_detection_boxes에 차원 추가
raw_detection_boxes = tensors['raw_detection_boxes']
raw_detection_scores = tensors['raw_detection_scores']

raw_detection_boxes_unsqueezed = gs.Variable('raw_detection_boxes_unsqueezed', dtype=np.float32)
unsqueeze_node = gs.Node(
    op='Unsqueeze',
    name='unsqueeze_raw_detection_boxes',
    attrs={
        'axes': [2]
    },
    inputs=[raw_detection_boxes],
    outputs=[raw_detection_boxes_unsqueezed],
)
graph.nodes.append(unsqueeze_node)

# nms 노드 추가
num_detections = gs.Variable('num_detections', dtype=np.int32, shape=(1, 1))
nmsed_boxes = gs.Variable('nmsed_boxes', dtype=np.float32, shape=(1, 100, 4))
nmsed_scores = gs.Variable('nmsed_scores', dtype=np.float32, shape=(1, 100))
nmsed_classes = gs.Variable('nmsed_classes', dtype=np.float32, shape=(1, 100))

nms_node = gs.Node(
    op='BatchedNMS_TRT',
    name='nms',
    attrs={
        "shareLocation": True, # 같은 박스로 모든 클래스에 대해 nms를 수행
        "numClasses": 6,
        "backgroundLabelId": -1, # 백그라운드 인덱스. 없는 경우 -1로 설정
        "topK": 4096,  # 스코어 순으로 박스를 정렬하여 상위 4096개만 연산
        "keepTopK": 100,  # nms 결과 중 스코어순으로 100개만 취함
        "scoreThreshold": 1e-8,
        "iouThreshold": 0.5,
        "isNormalized": True,  # 박스가 0~1 범위인 경우 True, 픽셀값이면 False
        "clipBoxes": True,  # 박스를 0~1 범위로 clip
        "scoreBits": 10,  # 스코어 비트 수. 높으면 nms 성능이 높은 대신 느려진다.
    },
    inputs=[raw_detection_boxes_unsqueezed, raw_detection_scores],
    outputs=[num_detections, nmsed_boxes, nmsed_scores, nmsed_classes],
)
graph.nodes.append(nms_node)

# 그래프의 아웃풋을 새로 정의
graph.outputs = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]
# clearup: 아웃풋에 관여하지 않는 노드를 제거한다.
# toposort: 그래프의 노드들을 순서에 맞게 자동 정렬한다.
graph.cleanup().toposort()
onnx.save_model(gs.export_onnx(graph), 'effdet_modify.onnx')