
Z
input_producer/ConstConst*
dtype0*.
value%B#B./dataset/train.tfrecords
=
input_producer/SizeConst*
dtype0*
value	B :
B
input_producer/Greater/yConst*
dtype0*
value	B : 
Y
input_producer/GreaterGreaterinput_producer/Sizeinput_producer/Greater/y*
T0
z
input_producer/Assert/ConstConst*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor
�
#input_producer/Assert/Assert/data_0Const*
dtype0*G
value>B< B6string_input_producer requires a non-null input tensor
�
input_producer/Assert/AssertAssertinput_producer/Greater#input_producer/Assert/Assert/data_0*
	summarize*

T
2
a
input_producer/IdentityIdentityinput_producer/Const^input_producer/Assert/Assert*
T0
h
input_producer/RandomShuffleRandomShuffleinput_producer/Identity*
seed2 *

seed *
T0
{
input_producerFIFOQueueV2*
shapes
: *
capacity *
shared_name *
	container *
component_types
2
�
)input_producer/input_producer_EnqueueManyQueueEnqueueManyV2input_producerinput_producer/RandomShuffle*

timeout_ms���������*
Tcomponents
2
b
#input_producer/input_producer_CloseQueueCloseV2input_producer*
cancel_pending_enqueues( 
d
%input_producer/input_producer_Close_1QueueCloseV2input_producer*
cancel_pending_enqueues(
A
"input_producer/input_producer_SizeQueueSizeV2input_producer
W
input_producer/CastCast"input_producer/input_producer_Size*

DstT0*

SrcT0
A
input_producer/mul/yConst*
dtype0*
valueB
 *   =
M
input_producer/mulMulinput_producer/Castinput_producer/mul/y*
T0
r
'input_producer/fraction_of_32_full/tagsConst*
dtype0*3
value*B( B"input_producer/fraction_of_32_full
y
"input_producer/fraction_of_32_fullScalarSummary'input_producer/fraction_of_32_full/tagsinput_producer/mul*
T0
`
TFRecordReaderV2TFRecordReaderV2*
	container *
shared_name *
compression_type 
>
ReaderReadV2ReaderReadV2TFRecordReaderV2input_producer
K
!ParseSingleExample/ExpandDims/dimConst*
dtype0*
value	B : 
s
ParseSingleExample/ExpandDims
ExpandDimsReaderReadV2:1!ParseSingleExample/ExpandDims/dim*

Tdim0*
T0
N
%ParseSingleExample/ParseExample/ConstConst*
dtype0*
valueB 
P
'ParseSingleExample/ParseExample/Const_1Const*
dtype0	*
valueB	 
[
2ParseSingleExample/ParseExample/ParseExample/namesConst*
dtype0*
valueB 
m
9ParseSingleExample/ParseExample/ParseExample/dense_keys_0Const*
dtype0*
valueB Btrain/image
m
9ParseSingleExample/ParseExample/ParseExample/dense_keys_1Const*
dtype0*
valueB Btrain/label
�
,ParseSingleExample/ParseExample/ParseExampleParseExampleParseSingleExample/ExpandDims2ParseSingleExample/ParseExample/ParseExample/names9ParseSingleExample/ParseExample/ParseExample/dense_keys_09ParseSingleExample/ParseExample/ParseExample/dense_keys_1%ParseSingleExample/ParseExample/Const'ParseSingleExample/ParseExample/Const_1*
sparse_types
 *
Tdense
2	*
Nsparse *
dense_shapes
: : *
Ndense

&ParseSingleExample/Squeeze_train/imageSqueeze,ParseSingleExample/ParseExample/ParseExample*
squeeze_dims
 *
T0
�
&ParseSingleExample/Squeeze_train/labelSqueeze.ParseSingleExample/ParseExample/ParseExample:1*
squeeze_dims
 *
T0	
c
	DecodeRaw	DecodeRaw&ParseSingleExample/Squeeze_train/image*
out_type0*
little_endian(
L
CastCast&ParseSingleExample/Squeeze_train/label*

DstT0*

SrcT0	
F
Reshape/shapeConst*
dtype0*!
valueB"           
C
ReshapeReshape	DecodeRawReshape/shape*
T0*
Tshape0
=
shuffle_batch/ConstConst*
dtype0
*
value	B
 Z
�
"shuffle_batch/random_shuffle_queueRandomShuffleQueueV2*
capacity*
component_types
2*
min_after_dequeue
*
shapes
:  : *
seed2 *

seed *
	container *
shared_name 
�
*shuffle_batch/random_shuffle_queue_enqueueQueueEnqueueV2"shuffle_batch/random_shuffle_queueReshapeCast*

timeout_ms���������*
Tcomponents
2
{
(shuffle_batch/random_shuffle_queue_CloseQueueCloseV2"shuffle_batch/random_shuffle_queue*
cancel_pending_enqueues( 
}
*shuffle_batch/random_shuffle_queue_Close_1QueueCloseV2"shuffle_batch/random_shuffle_queue*
cancel_pending_enqueues(
Z
'shuffle_batch/random_shuffle_queue_SizeQueueSizeV2"shuffle_batch/random_shuffle_queue
=
shuffle_batch/sub/yConst*
dtype0*
value	B :

_
shuffle_batch/subSub'shuffle_batch/random_shuffle_queue_Sizeshuffle_batch/sub/y*
T0
A
shuffle_batch/Maximum/xConst*
dtype0*
value	B : 
U
shuffle_batch/MaximumMaximumshuffle_batch/Maximum/xshuffle_batch/sub*
T0
I
shuffle_batch/CastCastshuffle_batch/Maximum*

DstT0*

SrcT0
@
shuffle_batch/mul/yConst*
dtype0*
valueB
 *��L=
J
shuffle_batch/mulMulshuffle_batch/Castshuffle_batch/mul/y*
T0
�
.shuffle_batch/fraction_over_10_of_20_full/tagsConst*
dtype0*:
value1B/ B)shuffle_batch/fraction_over_10_of_20_full
�
)shuffle_batch/fraction_over_10_of_20_fullScalarSummary.shuffle_batch/fraction_over_10_of_20_full/tagsshuffle_batch/mul*
T0
9
shuffle_batch/nConst*
dtype0*
value	B :

�
shuffle_batchQueueDequeueManyV2"shuffle_batch/random_shuffle_queueshuffle_batch/n*

timeout_ms���������*
component_types
2
=
Reshape_1/shapeConst*
dtype0*
valueB:

M
	Reshape_1Reshapeshuffle_batch:1Reshape_1/shape*
T0*
Tshape0
G
InputPlaceholder*
dtype0*$
shape:���������  
1
LabelsPlaceholder*
dtype0*
shape:
�
.conv2d/kernel/Initializer/random_uniform/shapeConst*
dtype0* 
_class
loc:@conv2d/kernel*%
valueB"            
{
,conv2d/kernel/Initializer/random_uniform/minConst*
dtype0* 
_class
loc:@conv2d/kernel*
valueB
 *�7'�
{
,conv2d/kernel/Initializer/random_uniform/maxConst*
dtype0* 
_class
loc:@conv2d/kernel*
valueB
 *�7'>
�
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0* 
_class
loc:@conv2d/kernel
�
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*
T0
�
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub* 
_class
loc:@conv2d/kernel*
T0
�
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*
T0
�
conv2d/kernel
VariableV2*
dtype0*
shape:*
	container *
shared_name * 
_class
loc:@conv2d/kernel
�
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
validate_shape(* 
_class
loc:@conv2d/kernel*
use_locking(*
T0
X
conv2d/kernel/readIdentityconv2d/kernel* 
_class
loc:@conv2d/kernel*
T0
n
conv2d/bias/Initializer/zerosConst*
dtype0*
_class
loc:@conv2d/bias*
valueB*    
{
conv2d/bias
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_class
loc:@conv2d/bias
�
conv2d/bias/AssignAssignconv2d/biasconv2d/bias/Initializer/zeros*
validate_shape(*
_class
loc:@conv2d/bias*
use_locking(*
T0
R
conv2d/bias/readIdentityconv2d/bias*
_class
loc:@conv2d/bias*
T0
U
conv2d/convolution/ShapeConst*
dtype0*%
valueB"            
U
 conv2d/convolution/dilation_rateConst*
dtype0*
valueB"      
�
conv2d/convolutionConv2DInputconv2d/kernel/read*
paddingVALID*
strides
*
use_cudnn_on_gpu(*
data_formatNHWC*
T0
_
conv2d/BiasAddBiasAddconv2d/convolutionconv2d/bias/read*
T0*
data_formatNHWC
,
conv2d/ReluReluconv2d/BiasAdd*
T0
�
max_pooling2d/MaxPoolMaxPoolconv2d/Relu*
paddingVALID*
strides
*
data_formatNHWC*
ksize
*
T0
�
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*"
_class
loc:@conv2d_1/kernel*%
valueB"            

.conv2d_1/kernel/Initializer/random_uniform/minConst*
dtype0*"
_class
loc:@conv2d_1/kernel*
valueB
 *�ս

.conv2d_1/kernel/Initializer/random_uniform/maxConst*
dtype0*"
_class
loc:@conv2d_1/kernel*
valueB
 *��=
�
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*"
_class
loc:@conv2d_1/kernel
�
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*
T0
�
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_1/kernel*
T0
�
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*
T0
�
conv2d_1/kernel
VariableV2*
dtype0*
shape:*
	container *
shared_name *"
_class
loc:@conv2d_1/kernel
�
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
use_locking(*
T0
^
conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
T0
r
conv2d_1/bias/Initializer/zerosConst*
dtype0* 
_class
loc:@conv2d_1/bias*
valueB*    

conv2d_1/bias
VariableV2*
dtype0*
shape:*
	container *
shared_name * 
_class
loc:@conv2d_1/bias
�
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/bias/Initializer/zeros*
validate_shape(* 
_class
loc:@conv2d_1/bias*
use_locking(*
T0
X
conv2d_1/bias/readIdentityconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
T0
W
conv2d_2/convolution/ShapeConst*
dtype0*%
valueB"            
W
"conv2d_2/convolution/dilation_rateConst*
dtype0*
valueB"      
�
conv2d_2/convolutionConv2Dmax_pooling2d/MaxPoolconv2d_1/kernel/read*
paddingVALID*
strides
*
use_cudnn_on_gpu(*
data_formatNHWC*
T0
e
conv2d_2/BiasAddBiasAddconv2d_2/convolutionconv2d_1/bias/read*
T0*
data_formatNHWC
0
conv2d_2/ReluReluconv2d_2/BiasAdd*
T0
�
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu*
paddingVALID*
strides
*
data_formatNHWC*
ksize
*
T0
D
Reshape_2/shapeConst*
dtype0*
valueB"�����  
U
	Reshape_2Reshapemax_pooling2d_2/MaxPoolReshape_2/shape*
T0*
Tshape0
0
dropout/IdentityIdentity	Reshape_2*
T0
�
-dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_class
loc:@dense/kernel*
valueB"�  x   
y
+dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_class
loc:@dense/kernel*
valueB
 *��۽
y
+dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_class
loc:@dense/kernel*
valueB
 *���=
�
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*
_class
loc:@dense/kernel
�
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
T0
�
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_class
loc:@dense/kernel*
T0
�
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
T0
�
dense/kernel
VariableV2*
dtype0*
shape:	�x*
	container *
shared_name *
_class
loc:@dense/kernel
�
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
validate_shape(*
_class
loc:@dense/kernel*
use_locking(*
T0
U
dense/kernel/readIdentitydense/kernel*
_class
loc:@dense/kernel*
T0
l
dense/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense/bias*
valueBx*    
y

dense/bias
VariableV2*
dtype0*
shape:x*
	container *
shared_name *
_class
loc:@dense/bias
�
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense/bias*
use_locking(*
T0
O
dense/bias/readIdentity
dense/bias*
_class
loc:@dense/bias*
T0
j
dense/MatMulMatMuldropout/Identitydense/kernel/read*
transpose_b( *
transpose_a( *
T0
W
dense/BiasAddBiasAdddense/MatMuldense/bias/read*
T0*
data_formatNHWC
*

dense/ReluReludense/BiasAdd*
T0
�
/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*!
_class
loc:@dense_1/kernel*
valueB"x   T   
}
-dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*!
_class
loc:@dense_1/kernel*
valueB
 *S�/�
}
-dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*!
_class
loc:@dense_1/kernel*
valueB
 *S�/>
�
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*!
_class
loc:@dense_1/kernel
�
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
T0
�
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel*
T0
�
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
T0
�
dense_1/kernel
VariableV2*
dtype0*
shape
:xT*
	container *
shared_name *!
_class
loc:@dense_1/kernel
�
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0
[
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
T0
p
dense_1/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense_1/bias*
valueBT*    
}
dense_1/bias
VariableV2*
dtype0*
shape:T*
	container *
shared_name *
_class
loc:@dense_1/bias
�
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense_1/bias*
use_locking(*
T0
U
dense_1/bias/readIdentitydense_1/bias*
_class
loc:@dense_1/bias*
T0
h
dense_2/MatMulMatMul
dense/Reludense_1/kernel/read*
transpose_b( *
transpose_a( *
T0
]
dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*
T0*
data_formatNHWC
.
dense_2/ReluReludense_2/BiasAdd*
T0
�
/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*!
_class
loc:@dense_2/kernel*
valueB"T      
}
-dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*!
_class
loc:@dense_2/kernel*
valueB
 *�<��
}
-dense_2/kernel/Initializer/random_uniform/maxConst*
dtype0*!
_class
loc:@dense_2/kernel*
valueB
 *�<�>
�
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*!
_class
loc:@dense_2/kernel
�
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_2/kernel*
T0
�
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_2/kernel*
T0
�
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_2/kernel*
T0
�
dense_2/kernel
VariableV2*
dtype0*
shape
:T*
	container *
shared_name *!
_class
loc:@dense_2/kernel
�
dense_2/kernel/AssignAssigndense_2/kernel)dense_2/kernel/Initializer/random_uniform*
validate_shape(*!
_class
loc:@dense_2/kernel*
use_locking(*
T0
[
dense_2/kernel/readIdentitydense_2/kernel*!
_class
loc:@dense_2/kernel*
T0
p
dense_2/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense_2/bias*
valueB*    
}
dense_2/bias
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_class
loc:@dense_2/bias
�
dense_2/bias/AssignAssigndense_2/biasdense_2/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense_2/bias*
use_locking(*
T0
U
dense_2/bias/readIdentitydense_2/bias*
_class
loc:@dense_2/bias*
T0
j
dense_3/MatMulMatMuldense_2/Reludense_2/kernel/read*
transpose_b( *
transpose_a( *
T0
]
dense_3/BiasAddBiasAdddense_3/MatMuldense_2/bias/read*
T0*
data_formatNHWC
5
output/yConst*
dtype0*
valueB
 *  �?
1
outputMuldense_3/BiasAddoutput/y*
T0
S
)SparseSoftmaxCrossEntropyWithLogits/ShapeShapeLabels*
out_type0*
T0
�
GSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsoutputLabels*
T0*
Tlabels0
3
ConstConst*
dtype0*
valueB: 
�
MeanMeanGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsConst*
T0*
	keep_dims( *

Tidx0
8
gradients/ShapeConst*
dtype0*
valueB 
<
gradients/ConstConst*
dtype0*
valueB
 *  �?
A
gradients/FillFillgradients/Shapegradients/Const*
T0
O
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
valueB:
p
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0
�
gradients/Mean_grad/ShapeShapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
out_type0*
T0
s
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0
�
gradients/Mean_grad/Shape_1ShapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
out_type0*
T0
D
gradients/Mean_grad/Shape_2Const*
dtype0*
valueB 
G
gradients/Mean_grad/ConstConst*
dtype0*
valueB: 
~
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
	keep_dims( *

Tidx0
I
gradients/Mean_grad/Const_1Const*
dtype0*
valueB: 
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
	keep_dims( *

Tidx0
G
gradients/Mean_grad/Maximum/yConst*
dtype0*
value	B :
j
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0
h
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0
V
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*

SrcT0
c
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0
u
gradients/zeros_like	ZerosLikeISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0
�
fgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*�
message��Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0
�
egradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
valueB :
���������
�
agradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsgradients/Mean_grad/truedivegradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0
�
Zgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulagradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsfgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0
N
gradients/output_grad/ShapeShapedense_3/BiasAdd*
out_type0*
T0
F
gradients/output_grad/Shape_1Const*
dtype0*
valueB 
�
+gradients/output_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/output_grad/Shapegradients/output_grad/Shape_1*
T0
�
gradients/output_grad/mulMulZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/muloutput/y*
T0
�
gradients/output_grad/SumSumgradients/output_grad/mul+gradients/output_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
w
gradients/output_grad/ReshapeReshapegradients/output_grad/Sumgradients/output_grad/Shape*
T0*
Tshape0
�
gradients/output_grad/mul_1Muldense_3/BiasAddZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul*
T0
�
gradients/output_grad/Sum_1Sumgradients/output_grad/mul_1-gradients/output_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
}
gradients/output_grad/Reshape_1Reshapegradients/output_grad/Sum_1gradients/output_grad/Shape_1*
T0*
Tshape0
p
&gradients/output_grad/tuple/group_depsNoOp^gradients/output_grad/Reshape ^gradients/output_grad/Reshape_1
�
.gradients/output_grad/tuple/control_dependencyIdentitygradients/output_grad/Reshape'^gradients/output_grad/tuple/group_deps*0
_class&
$"loc:@gradients/output_grad/Reshape*
T0
�
0gradients/output_grad/tuple/control_dependency_1Identitygradients/output_grad/Reshape_1'^gradients/output_grad/tuple/group_deps*2
_class(
&$loc:@gradients/output_grad/Reshape_1*
T0
�
*gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/output_grad/tuple/control_dependency*
T0*
data_formatNHWC
�
/gradients/dense_3/BiasAdd_grad/tuple/group_depsNoOp/^gradients/output_grad/tuple/control_dependency+^gradients/dense_3/BiasAdd_grad/BiasAddGrad
�
7gradients/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/output_grad/tuple/control_dependency0^gradients/dense_3/BiasAdd_grad/tuple/group_deps*0
_class&
$"loc:@gradients/output_grad/Reshape*
T0
�
9gradients/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/dense_3/BiasAdd_grad/BiasAddGrad0^gradients/dense_3/BiasAdd_grad/tuple/group_deps*=
_class3
1/loc:@gradients/dense_3/BiasAdd_grad/BiasAddGrad*
T0
�
$gradients/dense_3/MatMul_grad/MatMulMatMul7gradients/dense_3/BiasAdd_grad/tuple/control_dependencydense_2/kernel/read*
transpose_b(*
transpose_a( *
T0
�
&gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Relu7gradients/dense_3/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0
�
.gradients/dense_3/MatMul_grad/tuple/group_depsNoOp%^gradients/dense_3/MatMul_grad/MatMul'^gradients/dense_3/MatMul_grad/MatMul_1
�
6gradients/dense_3/MatMul_grad/tuple/control_dependencyIdentity$gradients/dense_3/MatMul_grad/MatMul/^gradients/dense_3/MatMul_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dense_3/MatMul_grad/MatMul*
T0
�
8gradients/dense_3/MatMul_grad/tuple/control_dependency_1Identity&gradients/dense_3/MatMul_grad/MatMul_1/^gradients/dense_3/MatMul_grad/tuple/group_deps*9
_class/
-+loc:@gradients/dense_3/MatMul_grad/MatMul_1*
T0

$gradients/dense_2/Relu_grad/ReluGradReluGrad6gradients/dense_3/MatMul_grad/tuple/control_dependencydense_2/Relu*
T0

*gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/dense_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC
�
/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp%^gradients/dense_2/Relu_grad/ReluGrad+^gradients/dense_2/BiasAdd_grad/BiasAddGrad
�
7gradients/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity$gradients/dense_2/Relu_grad/ReluGrad0^gradients/dense_2/BiasAdd_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dense_2/Relu_grad/ReluGrad*
T0
�
9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/dense_2/BiasAdd_grad/BiasAddGrad0^gradients/dense_2/BiasAdd_grad/tuple/group_deps*=
_class3
1/loc:@gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0
�
$gradients/dense_2/MatMul_grad/MatMulMatMul7gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_b(*
transpose_a( *
T0
�
&gradients/dense_2/MatMul_grad/MatMul_1MatMul
dense/Relu7gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0
�
.gradients/dense_2/MatMul_grad/tuple/group_depsNoOp%^gradients/dense_2/MatMul_grad/MatMul'^gradients/dense_2/MatMul_grad/MatMul_1
�
6gradients/dense_2/MatMul_grad/tuple/control_dependencyIdentity$gradients/dense_2/MatMul_grad/MatMul/^gradients/dense_2/MatMul_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dense_2/MatMul_grad/MatMul*
T0
�
8gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity&gradients/dense_2/MatMul_grad/MatMul_1/^gradients/dense_2/MatMul_grad/tuple/group_deps*9
_class/
-+loc:@gradients/dense_2/MatMul_grad/MatMul_1*
T0
{
"gradients/dense/Relu_grad/ReluGradReluGrad6gradients/dense_2/MatMul_grad/tuple/control_dependency
dense/Relu*
T0
{
(gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC
�
-gradients/dense/BiasAdd_grad/tuple/group_depsNoOp#^gradients/dense/Relu_grad/ReluGrad)^gradients/dense/BiasAdd_grad/BiasAddGrad
�
5gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentity"gradients/dense/Relu_grad/ReluGrad.^gradients/dense/BiasAdd_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dense/Relu_grad/ReluGrad*
T0
�
7gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/dense/BiasAdd_grad/BiasAddGrad.^gradients/dense/BiasAdd_grad/tuple/group_deps*;
_class1
/-loc:@gradients/dense/BiasAdd_grad/BiasAddGrad*
T0
�
"gradients/dense/MatMul_grad/MatMulMatMul5gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
transpose_b(*
transpose_a( *
T0
�
$gradients/dense/MatMul_grad/MatMul_1MatMuldropout/Identity5gradients/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0
�
,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
�
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul*
T0
�
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1*
T0
Y
gradients/Reshape_2_grad/ShapeShapemax_pooling2d_2/MaxPool*
out_type0*
T0
�
 gradients/Reshape_2_grad/ReshapeReshape4gradients/dense/MatMul_grad/tuple/control_dependencygradients/Reshape_2_grad/Shape*
T0*
Tshape0
�
2gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d_2/Relumax_pooling2d_2/MaxPool gradients/Reshape_2_grad/Reshape*
paddingVALID*
strides
*
ksize
*
data_formatNHWC*
T0
}
%gradients/conv2d_2/Relu_grad/ReluGradReluGrad2gradients/max_pooling2d_2/MaxPool_grad/MaxPoolGradconv2d_2/Relu*
T0
�
+gradients/conv2d_2/BiasAdd_grad/BiasAddGradBiasAddGrad%gradients/conv2d_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC
�
0gradients/conv2d_2/BiasAdd_grad/tuple/group_depsNoOp&^gradients/conv2d_2/Relu_grad/ReluGrad,^gradients/conv2d_2/BiasAdd_grad/BiasAddGrad
�
8gradients/conv2d_2/BiasAdd_grad/tuple/control_dependencyIdentity%gradients/conv2d_2/Relu_grad/ReluGrad1^gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*8
_class.
,*loc:@gradients/conv2d_2/Relu_grad/ReluGrad*
T0
�
:gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1Identity+gradients/conv2d_2/BiasAdd_grad/BiasAddGrad1^gradients/conv2d_2/BiasAdd_grad/tuple/group_deps*>
_class4
20loc:@gradients/conv2d_2/BiasAdd_grad/BiasAddGrad*
T0
b
)gradients/conv2d_2/convolution_grad/ShapeShapemax_pooling2d/MaxPool*
out_type0*
T0
�
7gradients/conv2d_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInput)gradients/conv2d_2/convolution_grad/Shapeconv2d_1/kernel/read8gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
strides
*
use_cudnn_on_gpu(*
data_formatNHWC*
T0
h
+gradients/conv2d_2/convolution_grad/Shape_1Const*
dtype0*%
valueB"            
�
8gradients/conv2d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool+gradients/conv2d_2/convolution_grad/Shape_18gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
strides
*
use_cudnn_on_gpu(*
data_formatNHWC*
T0
�
4gradients/conv2d_2/convolution_grad/tuple/group_depsNoOp8^gradients/conv2d_2/convolution_grad/Conv2DBackpropInput9^gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter
�
<gradients/conv2d_2/convolution_grad/tuple/control_dependencyIdentity7gradients/conv2d_2/convolution_grad/Conv2DBackpropInput5^gradients/conv2d_2/convolution_grad/tuple/group_deps*J
_class@
><loc:@gradients/conv2d_2/convolution_grad/Conv2DBackpropInput*
T0
�
>gradients/conv2d_2/convolution_grad/tuple/control_dependency_1Identity8gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter5^gradients/conv2d_2/convolution_grad/tuple/group_deps*K
_classA
?=loc:@gradients/conv2d_2/convolution_grad/Conv2DBackpropFilter*
T0
�
0gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d/Relumax_pooling2d/MaxPool<gradients/conv2d_2/convolution_grad/tuple/control_dependency*
paddingVALID*
strides
*
ksize
*
data_formatNHWC*
T0
w
#gradients/conv2d/Relu_grad/ReluGradReluGrad0gradients/max_pooling2d/MaxPool_grad/MaxPoolGradconv2d/Relu*
T0
}
)gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients/conv2d/Relu_grad/ReluGrad*
T0*
data_formatNHWC
�
.gradients/conv2d/BiasAdd_grad/tuple/group_depsNoOp$^gradients/conv2d/Relu_grad/ReluGrad*^gradients/conv2d/BiasAdd_grad/BiasAddGrad
�
6gradients/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity#gradients/conv2d/Relu_grad/ReluGrad/^gradients/conv2d/BiasAdd_grad/tuple/group_deps*6
_class,
*(loc:@gradients/conv2d/Relu_grad/ReluGrad*
T0
�
8gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity)gradients/conv2d/BiasAdd_grad/BiasAddGrad/^gradients/conv2d/BiasAdd_grad/tuple/group_deps*<
_class2
0.loc:@gradients/conv2d/BiasAdd_grad/BiasAddGrad*
T0
P
'gradients/conv2d/convolution_grad/ShapeShapeInput*
out_type0*
T0
�
5gradients/conv2d/convolution_grad/Conv2DBackpropInputConv2DBackpropInput'gradients/conv2d/convolution_grad/Shapeconv2d/kernel/read6gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
strides
*
use_cudnn_on_gpu(*
data_formatNHWC*
T0
f
)gradients/conv2d/convolution_grad/Shape_1Const*
dtype0*%
valueB"            
�
6gradients/conv2d/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterInput)gradients/conv2d/convolution_grad/Shape_16gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
paddingVALID*
strides
*
use_cudnn_on_gpu(*
data_formatNHWC*
T0
�
2gradients/conv2d/convolution_grad/tuple/group_depsNoOp6^gradients/conv2d/convolution_grad/Conv2DBackpropInput7^gradients/conv2d/convolution_grad/Conv2DBackpropFilter
�
:gradients/conv2d/convolution_grad/tuple/control_dependencyIdentity5gradients/conv2d/convolution_grad/Conv2DBackpropInput3^gradients/conv2d/convolution_grad/tuple/group_deps*H
_class>
<:loc:@gradients/conv2d/convolution_grad/Conv2DBackpropInput*
T0
�
<gradients/conv2d/convolution_grad/tuple/control_dependency_1Identity6gradients/conv2d/convolution_grad/Conv2DBackpropFilter3^gradients/conv2d/convolution_grad/tuple/group_deps*I
_class?
=;loc:@gradients/conv2d/convolution_grad/Conv2DBackpropFilter*
T0
J
GradientDescent/learning_rateConst*
dtype0*
valueB
 *o�:
�
9GradientDescent/update_conv2d/kernel/ApplyGradientDescentApplyGradientDescentconv2d/kernelGradientDescent/learning_rate<gradients/conv2d/convolution_grad/tuple/control_dependency_1* 
_class
loc:@conv2d/kernel*
use_locking( *
T0
�
7GradientDescent/update_conv2d/bias/ApplyGradientDescentApplyGradientDescentconv2d/biasGradientDescent/learning_rate8gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@conv2d/bias*
use_locking( *
T0
�
;GradientDescent/update_conv2d_1/kernel/ApplyGradientDescentApplyGradientDescentconv2d_1/kernelGradientDescent/learning_rate>gradients/conv2d_2/convolution_grad/tuple/control_dependency_1*"
_class
loc:@conv2d_1/kernel*
use_locking( *
T0
�
9GradientDescent/update_conv2d_1/bias/ApplyGradientDescentApplyGradientDescentconv2d_1/biasGradientDescent/learning_rate:gradients/conv2d_2/BiasAdd_grad/tuple/control_dependency_1* 
_class
loc:@conv2d_1/bias*
use_locking( *
T0
�
8GradientDescent/update_dense/kernel/ApplyGradientDescentApplyGradientDescentdense/kernelGradientDescent/learning_rate6gradients/dense/MatMul_grad/tuple/control_dependency_1*
_class
loc:@dense/kernel*
use_locking( *
T0
�
6GradientDescent/update_dense/bias/ApplyGradientDescentApplyGradientDescent
dense/biasGradientDescent/learning_rate7gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense/bias*
use_locking( *
T0
�
:GradientDescent/update_dense_1/kernel/ApplyGradientDescentApplyGradientDescentdense_1/kernelGradientDescent/learning_rate8gradients/dense_2/MatMul_grad/tuple/control_dependency_1*!
_class
loc:@dense_1/kernel*
use_locking( *
T0
�
8GradientDescent/update_dense_1/bias/ApplyGradientDescentApplyGradientDescentdense_1/biasGradientDescent/learning_rate9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_1/bias*
use_locking( *
T0
�
:GradientDescent/update_dense_2/kernel/ApplyGradientDescentApplyGradientDescentdense_2/kernelGradientDescent/learning_rate8gradients/dense_3/MatMul_grad/tuple/control_dependency_1*!
_class
loc:@dense_2/kernel*
use_locking( *
T0
�
8GradientDescent/update_dense_2/bias/ApplyGradientDescentApplyGradientDescentdense_2/biasGradientDescent/learning_rate9gradients/dense_3/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@dense_2/bias*
use_locking( *
T0
�
GradientDescentNoOp:^GradientDescent/update_conv2d/kernel/ApplyGradientDescent8^GradientDescent/update_conv2d/bias/ApplyGradientDescent<^GradientDescent/update_conv2d_1/kernel/ApplyGradientDescent:^GradientDescent/update_conv2d_1/bias/ApplyGradientDescent9^GradientDescent/update_dense/kernel/ApplyGradientDescent7^GradientDescent/update_dense/bias/ApplyGradientDescent;^GradientDescent/update_dense_1/kernel/ApplyGradientDescent9^GradientDescent/update_dense_1/bias/ApplyGradientDescent;^GradientDescent/update_dense_2/kernel/ApplyGradientDescent9^GradientDescent/update_dense_2/bias/ApplyGradientDescent
2
InTopKInTopKoutputLabels*
k*
T0
.
Cast_2CastInTopK*

DstT0*

SrcT0

5
Const_1Const*
dtype0*
valueB: 
G
accuracyMeanCast_2Const_1*
T0*
	keep_dims( *

Tidx0
�
initNoOp^conv2d/kernel/Assign^conv2d/bias/Assign^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^dense/kernel/Assign^dense/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign
8

save/ConstConst*
dtype0*
valueB Bmodel
�
save/SaveV2/tensor_namesConst*
dtype0*�
value�B�
Bconv2d/biasBconv2d/kernelBconv2d_1/biasBconv2d_1/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernel
[
save/SaveV2/shape_and_slicesConst*
dtype0*'
valueB
B B B B B B B B B B 
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesconv2d/biasconv2d/kernelconv2d_1/biasconv2d_1/kernel
dense/biasdense/kerneldense_1/biasdense_1/kerneldense_2/biasdense_2/kernel*
dtypes
2

e
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
T0
S
save/RestoreV2/tensor_namesConst*
dtype0* 
valueBBconv2d/bias
L
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B 
v
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2
�
save/AssignAssignconv2d/biassave/RestoreV2*
validate_shape(*
_class
loc:@conv2d/bias*
use_locking(*
T0
W
save/RestoreV2_1/tensor_namesConst*
dtype0*"
valueBBconv2d/kernel
N
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2
�
save/Assign_1Assignconv2d/kernelsave/RestoreV2_1*
validate_shape(* 
_class
loc:@conv2d/kernel*
use_locking(*
T0
W
save/RestoreV2_2/tensor_namesConst*
dtype0*"
valueBBconv2d_1/bias
N
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2
�
save/Assign_2Assignconv2d_1/biassave/RestoreV2_2*
validate_shape(* 
_class
loc:@conv2d_1/bias*
use_locking(*
T0
Y
save/RestoreV2_3/tensor_namesConst*
dtype0*$
valueBBconv2d_1/kernel
N
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2
�
save/Assign_3Assignconv2d_1/kernelsave/RestoreV2_3*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
use_locking(*
T0
T
save/RestoreV2_4/tensor_namesConst*
dtype0*
valueBB
dense/bias
N
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2
�
save/Assign_4Assign
dense/biassave/RestoreV2_4*
validate_shape(*
_class
loc:@dense/bias*
use_locking(*
T0
V
save/RestoreV2_5/tensor_namesConst*
dtype0*!
valueBBdense/kernel
N
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2
�
save/Assign_5Assigndense/kernelsave/RestoreV2_5*
validate_shape(*
_class
loc:@dense/kernel*
use_locking(*
T0
V
save/RestoreV2_6/tensor_namesConst*
dtype0*!
valueBBdense_1/bias
N
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2
�
save/Assign_6Assigndense_1/biassave/RestoreV2_6*
validate_shape(*
_class
loc:@dense_1/bias*
use_locking(*
T0
X
save/RestoreV2_7/tensor_namesConst*
dtype0*#
valueBBdense_1/kernel
N
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2
�
save/Assign_7Assigndense_1/kernelsave/RestoreV2_7*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0
V
save/RestoreV2_8/tensor_namesConst*
dtype0*!
valueBBdense_2/bias
N
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2
�
save/Assign_8Assigndense_2/biassave/RestoreV2_8*
validate_shape(*
_class
loc:@dense_2/bias*
use_locking(*
T0
X
save/RestoreV2_9/tensor_namesConst*
dtype0*#
valueBBdense_2/kernel
N
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2
�
save/Assign_9Assigndense_2/kernelsave/RestoreV2_9*
validate_shape(*!
_class
loc:@dense_2/kernel*
use_locking(*
T0
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9"