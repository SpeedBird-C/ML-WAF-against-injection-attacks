П
Ъ((
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
Ё
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
Ј
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
Г
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Ѕ
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
њ
UnicodeDecode	
input

row_splits"Tsplits
char_values"
input_encodingstring"8
errorsstring	replace:
strictreplaceignore"
replacement_charint§џ"&
replace_control_charactersbool( "
Tsplitstype0	:
2	
ј
UnicodeEncode
input_values
input_splits"Tsplits

output"8
errorsstring	replace:
ignorereplacestrict":
output_encodingstring:
UTF-8	UTF-16-BE	UTF-32-BE"
replacement_charint§џ"
Tsplitstype0	:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ѓ№

embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	н-@*%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	н-@*
dtype0
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
:@@*
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
:@*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
k

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name229*
value_dtype0	
~
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_106*
value_dtype0	
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0

Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	н-@*,
shared_nameAdam/embedding/embeddings/m

/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m*
_output_shapes
:	н-@*
dtype0

Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*%
shared_nameAdam/conv1d/kernel/m

(Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/m*"
_output_shapes
:@@*
dtype0
|
Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv1d/bias/m
u
&Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/m*
_output_shapes
:@*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0

Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	н-@*,
shared_nameAdam/embedding/embeddings/v

/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v*
_output_shapes
:	н-@*
dtype0

Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*%
shared_nameAdam/conv1d/kernel/v

(Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/v*"
_output_shapes
:@@*
dtype0
|
Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv1d/bias/v
u
&Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/v*
_output_shapes
:@*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
ъ
Const_4Const*
_output_shapes
:]*
dtype0*Ў
valueЄBЁ]BeB BsBaBrBtBlB_BnBcB)BoB(BiB.B'B/B1BuBdBhB0BmBpB5B2BbB3B6B\B*B4BxB,B=B|BgB7BfB}B{B8B]B[BwB"B9B;ByB-BkBvB#B>B<B+B&B%B!BSBqBNBCBTBjBIBEBOB:BUBRBLBzBABWBMB$BPBFBDBяПНBXBHBGBYBVB@BQBBB`BZBJB?
М
Const_5Const*
_output_shapes
:]*
dtype0	*
valueіBѓ	]"ш                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       

StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *"
fR
__inference_<lambda>_4959
ш
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *"
fR
__inference_<lambda>_4964
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
Ч
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
Ы?
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?
valueњ>Bї> B№>
Ї
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
;
_lookup_layer
	keras_api
_adapt_function*
Ё
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses* 
* 
'
!1
"2
#3
$4
%5*
'
!0
"1
#2
$3
%4*
* 
А
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

+serving_default* 
7
,lookup_table
-token_counts
.	keras_api*
* 
* 
 
!
embeddings
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses*
І

"kernel
#bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses*

;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses* 
І

$kernel
%bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses*
Ј
Giter

Hbeta_1

Ibeta_2
	Jdecay
Klearning_rate!m"m#m$m%m!v"v#v$v%v*
'
!0
"1
#2
$3
%4*
'
!0
"1
#2
$3
%4*
* 

Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses* 
* 
* 
TN
VARIABLE_VALUEembedding/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv1d/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv1d/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

V0
W1*
* 
* 
* 
R
X_initializer
Y_create_resource
Z_initialize
[_destroy_resource* 

\_create_resource
]_initialize
^_destroy_resourceJ
tableAlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table*
* 

!0*

!0*
* 

_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*
* 
* 

"0
#1*

"0
#1*
* 

dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses* 
* 
* 

$0
%1*

$0
%1*
* 

nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*
* 
* 
a[
VARIABLE_VALUE	Adam/iter>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/beta_1@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/beta_2@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE
Adam/decay?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/learning_rateGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

s0
t1*
* 
* 
* 
* 
* 
* 
* 
8
	utotal
	vcount
w	variables
x	keras_api*
H
	ytotal
	zcount
{
_fn_kwargs
|	variables
}	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
:
	~total
	count
	variables
	keras_api*
M

total

count

_fn_kwargs
	variables
	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

u0
v1*

w	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

y0
z1*

|	variables*
jd
VARIABLE_VALUEtotal_2Ilayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEcount_2Ilayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

~0
1*

	variables*
jd
VARIABLE_VALUEtotal_3Ilayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEcount_3Ilayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

	variables*

VARIABLE_VALUEAdam/embedding/embeddings/mWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv1d/kernel/mWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv1d/bias/mWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense/kernel/mWvariables/4/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense/bias/mWvariables/5/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/embedding/embeddings/vWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv1d/kernel/vWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv1d/bias/vWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense/kernel/vWvariables/4/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense/bias/vWvariables/5/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

(serving_default_text_vectorization_inputPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
Т
StatefulPartitionedCall_1StatefulPartitionedCall(serving_default_text_vectorization_input
hash_tableConstConst_1Const_2embedding/embeddingsconv1d/kernelconv1d/biasdense/kernel
dense/bias*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_4564
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
 
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename(embedding/embeddings/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1Adam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp(Adam/conv1d/kernel/m/Read/ReadVariableOp&Adam/conv1d/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp(Adam/conv1d/kernel/v/Read/ReadVariableOp&Adam/conv1d/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst_6*+
Tin$
"2 		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_5085
і
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameembedding/embeddingsconv1d/kernelconv1d/biasdense/kernel
dense/biasMutableHashTable	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2total_3count_3Adam/embedding/embeddings/mAdam/conv1d/kernel/mAdam/conv1d/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/embedding/embeddings/vAdam/conv1d/kernel/vAdam/conv1d/bias/vAdam/dense/kernel/vAdam/dense/bias/v*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_5182Ве
і
ђ
__inference__initializer_49046
2key_value_init228_lookuptableimportv2_table_handle.
*key_value_init228_lookuptableimportv2_keys0
,key_value_init228_lookuptableimportv2_values	
identityЂ%key_value_init228/LookupTableImportV2ї
%key_value_init228/LookupTableImportV2LookupTableImportV22key_value_init228_lookuptableimportv2_table_handle*key_value_init228_lookuptableimportv2_keys,key_value_init228_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: n
NoOpNoOp&^key_value_init228/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :]:]2N
%key_value_init228/LookupTableImportV2%key_value_init228/LookupTableImportV2: 

_output_shapes
:]: 

_output_shapes
:]


Е
"__inference_signature_wrapper_4564
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	н-@
	unknown_4:@@
	unknown_5:@
	unknown_6:@
	unknown_7:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_3617o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:џџџџџџџџџ
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
я
O
3__inference_global_max_pooling1d_layer_call_fn_4866

inputs
identityТ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_3627i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

+
__inference__destroyer_4924
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 


Ќ
+__inference_sequential_1_layer_call_fn_4363

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	н-@
	unknown_4:@@
	unknown_5:@
	unknown_6:@
	unknown_7:
identityЂStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_4129o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
#
Њ
D__inference_sequential_layer_call_and_return_conditional_losses_4809

inputs	2
embedding_embedding_lookup_4783:	н-@H
2conv1d_conv1d_expanddims_1_readvariableop_resource:@@4
&conv1d_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:
identityЂconv1d/BiasAdd/ReadVariableOpЂ)conv1d/Conv1D/ExpandDims_1/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂembedding/embedding_lookupV
CastCastinputs*

DstT0*

SrcT0	*(
_output_shapes
:џџџџџџџџџњb
embedding/CastCastCast:y:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџњр
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_4783embedding/Cast:y:0*
Tindices0*2
_class(
&$loc:@embedding/embedding_lookup/4783*,
_output_shapes
:џџџџџџџџџњ@*
dtype0П
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/4783*,
_output_shapes
:џџџџџџџџџњ@
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџИ
conv1d/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@ 
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Е
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@Т
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@l
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ё
global_max_pooling1d/MaxMaxconv1d/Relu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense/MatMulMatMul!global_max_pooling1d/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџe
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџь
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:P L
(
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
ЈA

__inference__traced_save_5085
file_prefix3
/savev2_embedding_embeddings_read_readvariableop,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableopJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop3
/savev2_adam_conv1d_kernel_m_read_readvariableop1
-savev2_adam_conv1d_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop3
/savev2_adam_conv1d_kernel_v_read_readvariableop1
-savev2_adam_conv1d_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const_6

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: А
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*й
valueЯBЬB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/4/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/5/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/4/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/5/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЋ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_embedding_embeddings_read_readvariableop(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableopFsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop/savev2_adam_conv1d_kernel_m_read_readvariableop-savev2_adam_conv1d_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop/savev2_adam_conv1d_kernel_v_read_readvariableop-savev2_adam_conv1d_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *-
dtypes#
!2		
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ш
_input_shapesЖ
Г: :	н-@:@@:@:@:::: : : : : : : : : : : : : :	н-@:@@:@:@::	н-@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	н-@:($
"
_output_shapes
:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	н-@:($
"
_output_shapes
:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	н-@:($
"
_output_shapes
:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
Ч

О
+__inference_sequential_1_layer_call_fn_4173
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	н-@
	unknown_4:@@
	unknown_5:@
	unknown_6:@
	unknown_7:
identityЂStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_4129o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:џџџџџџџџџ
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
#
Њ
D__inference_sequential_layer_call_and_return_conditional_losses_3924

inputs	2
embedding_embedding_lookup_3898:	н-@H
2conv1d_conv1d_expanddims_1_readvariableop_resource:@@4
&conv1d_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:
identityЂconv1d/BiasAdd/ReadVariableOpЂ)conv1d/Conv1D/ExpandDims_1/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂembedding/embedding_lookupV
CastCastinputs*

DstT0*

SrcT0	*(
_output_shapes
:џџџџџџџџџњb
embedding/CastCastCast:y:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџњр
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_3898embedding/Cast:y:0*
Tindices0*2
_class(
&$loc:@embedding/embedding_lookup/3898*,
_output_shapes
:џџџџџџџџџњ@*
dtype0П
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/3898*,
_output_shapes
:џџџџџџџџџњ@
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџИ
conv1d/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@ 
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Е
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@Т
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@l
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ё
global_max_pooling1d/MaxMaxconv1d/Relu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense/MatMulMatMul!global_max_pooling1d/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџe
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџь
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:P L
(
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs


Ќ
+__inference_sequential_1_layer_call_fn_4340

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	н-@
	unknown_4:@@
	unknown_5:@
	unknown_6:@
	unknown_7:
identityЂStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_3944o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
цГ
	
__inference__wrapped_model_3617
text_vectorization_input\
Xsequential_1_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle]
Ysequential_1_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	9
5sequential_1_text_vectorization_string_lookup_equal_y<
8sequential_1_text_vectorization_string_lookup_selectv2_t	J
7sequential_1_sequential_embedding_embedding_lookup_3590:	н-@`
Jsequential_1_sequential_conv1d_conv1d_expanddims_1_readvariableop_resource:@@L
>sequential_1_sequential_conv1d_biasadd_readvariableop_resource:@N
<sequential_1_sequential_dense_matmul_readvariableop_resource:@K
=sequential_1_sequential_dense_biasadd_readvariableop_resource:
identityЂ5sequential_1/sequential/conv1d/BiasAdd/ReadVariableOpЂAsequential_1/sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOpЂ4sequential_1/sequential/dense/BiasAdd/ReadVariableOpЂ3sequential_1/sequential/dense/MatMul/ReadVariableOpЂ2sequential_1/sequential/embedding/embedding_lookupЂKsequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2
:sequential_1/text_vectorization/UnicodeSplit/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџЬ
4sequential_1/text_vectorization/UnicodeSplit/ReshapeReshapetext_vectorization_inputCsequential_1/text_vectorization/UnicodeSplit/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџй
:sequential_1/text_vectorization/UnicodeSplit/UnicodeDecodeUnicodeDecode=sequential_1/text_vectorization/UnicodeSplit/Reshape:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
input_encodingUTF-8
Lsequential_1/text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Љ
Hsequential_1/text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims
ExpandDimsHsequential_1/text_vectorization/UnicodeSplit/UnicodeDecode:char_values:0Usequential_1/text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџ№
_sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ShapeShapeQsequential_1/text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	З
msequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Й
osequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Й
osequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_sliceStridedSlicehsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0vsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack:output:0xsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1:output:0xsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЙ
osequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
qsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
qsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
isequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1StridedSlicehsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0xsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack:output:0zsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1:output:0zsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЙ
osequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Л
qsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
qsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
isequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2StridedSlicehsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0xsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack:output:0zsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1:output:0zsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskэ
]sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mulMulrsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1:output:0rsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: Й
osequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:Л
qsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Л
qsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
isequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3StridedSlicehsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0xsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack:output:0zsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1:output:0zsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_mask
isequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0Packasequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:Ї
esequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ђ
`sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concatConcatV2rsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0:output:0rsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3:output:0nsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:ц
asequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ReshapeReshapeQsequential_1/text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims:output:0isequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat:output:0*
T0*
Tshape0	*#
_output_shapes
:џџџџџџџџџЙ
osequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
qsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
qsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
isequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4StridedSlicehsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0xsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack:output:0zsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1:output:0zsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЁ
_sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 RЄ
zsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/ShapeShapejsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0*
T0*
_output_shapes
:*
out_type0	г
sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: е
sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:е
sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSlicesequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskо
sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rп
sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV2rsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4:output:0Єsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: ф
Ёsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R ф
Ёsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 Rа
sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRangeЊsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0Њsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

Tidx0	*#
_output_shapes
:џџџџџџџџџр
sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMulЄsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0hsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Const:output:0*
T0	*#
_output_shapes
:џџџџџџџџџТ
dsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncodeUnicodeEncodejsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0sequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*#
_output_shapes
:џџџџџџџџџ*
output_encodingUTF-8Р
Ksequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Xsequential_1_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handlemsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0Ysequential_1_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:џџџџџџџџџ
3sequential_1/text_vectorization/string_lookup/EqualEqualmsequential_1/text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:05sequential_1_text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:џџџџџџџџџЙ
6sequential_1/text_vectorization/string_lookup/SelectV2SelectV27sequential_1/text_vectorization/string_lookup/Equal:z:08sequential_1_text_vectorization_string_lookup_selectv2_tTsequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:џџџџџџџџџБ
6sequential_1/text_vectorization/string_lookup/IdentityIdentity?sequential_1/text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ~
<sequential_1/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
4sequential_1/text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"џџџџџџџџњ       
Csequential_1/text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor=sequential_1/text_vectorization/RaggedToTensor/Const:output:0?sequential_1/text_vectorization/string_lookup/Identity:output:0Esequential_1/text_vectorization/RaggedToTensor/default_value:output:0Gsequential_1/text_vectorization/UnicodeSplit/UnicodeDecode:row_splits:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:џџџџџџџџџњ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITSД
sequential_1/sequential/CastCastLsequential_1/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*

DstT0*

SrcT0	*(
_output_shapes
:џџџџџџџџџњ
&sequential_1/sequential/embedding/CastCast sequential_1/sequential/Cast:y:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџњР
2sequential_1/sequential/embedding/embedding_lookupResourceGather7sequential_1_sequential_embedding_embedding_lookup_3590*sequential_1/sequential/embedding/Cast:y:0*
Tindices0*J
_class@
><loc:@sequential_1/sequential/embedding/embedding_lookup/3590*,
_output_shapes
:џџџџџџџџџњ@*
dtype0
;sequential_1/sequential/embedding/embedding_lookup/IdentityIdentity;sequential_1/sequential/embedding/embedding_lookup:output:0*
T0*J
_class@
><loc:@sequential_1/sequential/embedding/embedding_lookup/3590*,
_output_shapes
:џџџџџџџџџњ@Ц
=sequential_1/sequential/embedding/embedding_lookup/Identity_1IdentityDsequential_1/sequential/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@
4sequential_1/sequential/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
0sequential_1/sequential/conv1d/Conv1D/ExpandDims
ExpandDimsFsequential_1/sequential/embedding/embedding_lookup/Identity_1:output:0=sequential_1/sequential/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@а
Asequential_1/sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpJsequential_1_sequential_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0x
6sequential_1/sequential/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : §
2sequential_1/sequential/conv1d/Conv1D/ExpandDims_1
ExpandDimsIsequential_1/sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0?sequential_1/sequential/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@
%sequential_1/sequential/conv1d/Conv1DConv2D9sequential_1/sequential/conv1d/Conv1D/ExpandDims:output:0;sequential_1/sequential/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
П
-sequential_1/sequential/conv1d/Conv1D/SqueezeSqueeze.sequential_1/sequential/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџА
5sequential_1/sequential/conv1d/BiasAdd/ReadVariableOpReadVariableOp>sequential_1_sequential_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0п
&sequential_1/sequential/conv1d/BiasAddBiasAdd6sequential_1/sequential/conv1d/Conv1D/Squeeze:output:0=sequential_1/sequential/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@
#sequential_1/sequential/conv1d/ReluRelu/sequential_1/sequential/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@
Bsequential_1/sequential/global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :щ
0sequential_1/sequential/global_max_pooling1d/MaxMax1sequential_1/sequential/conv1d/Relu:activations:0Ksequential_1/sequential/global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@А
3sequential_1/sequential/dense/MatMul/ReadVariableOpReadVariableOp<sequential_1_sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0и
$sequential_1/sequential/dense/MatMulMatMul9sequential_1/sequential/global_max_pooling1d/Max:output:0;sequential_1/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЎ
4sequential_1/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0а
%sequential_1/sequential/dense/BiasAddBiasAdd.sequential_1/sequential/dense/MatMul:product:0<sequential_1/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
sequential_1/activation/SigmoidSigmoid.sequential_1/sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџr
IdentityIdentity#sequential_1/activation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџВ
NoOpNoOp6^sequential_1/sequential/conv1d/BiasAdd/ReadVariableOpB^sequential_1/sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp5^sequential_1/sequential/dense/BiasAdd/ReadVariableOp4^sequential_1/sequential/dense/MatMul/ReadVariableOp3^sequential_1/sequential/embedding/embedding_lookupL^sequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : : : : : 2n
5sequential_1/sequential/conv1d/BiasAdd/ReadVariableOp5sequential_1/sequential/conv1d/BiasAdd/ReadVariableOp2
Asequential_1/sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOpAsequential_1/sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2l
4sequential_1/sequential/dense/BiasAdd/ReadVariableOp4sequential_1/sequential/dense/BiasAdd/ReadVariableOp2j
3sequential_1/sequential/dense/MatMul/ReadVariableOp3sequential_1/sequential/dense/MatMul/ReadVariableOp2h
2sequential_1/sequential/embedding/embedding_lookup2sequential_1/sequential/embedding/embedding_lookup2
Ksequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Ksequential_1/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:џџџџџџџџџ
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ь
9
__inference__creator_4896
identityЂ
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name229*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table

-
__inference__initializer_4919
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
ф
ї
)__inference_sequential_layer_call_fn_3795
embedding_input
unknown:	н-@
	unknown_0:@@
	unknown_1:@
	unknown_2:@
	unknown_3:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_3767o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:џџџџџџџџџњ
)
_user_specified_nameembedding_input
Т	
№
?__inference_dense_layer_call_and_return_conditional_losses_4891

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

с
D__inference_sequential_layer_call_and_return_conditional_losses_3691

inputs!
embedding_3648:	н-@!
conv1d_3668:@@
conv1d_3670:@

dense_3685:@

dense_3687:
identityЂconv1d/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂ!embedding/StatefulPartitionedCallс
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_3648*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_3647
conv1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_3668conv1d_3670*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_3667я
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_3627
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0
dense_3685
dense_3687*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_3684u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЋ
NoOpNoOp^conv1d/StatefulPartitionedCall^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Щ
ю
)__inference_sequential_layer_call_fn_4687

inputs	
unknown:	н-@
	unknown_0:@@
	unknown_1:@
	unknown_2:@
	unknown_3:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4019o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Ы

@__inference_conv1d_layer_call_and_return_conditional_losses_3667

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџњ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
Г

F__inference_sequential_1_layer_call_and_return_conditional_losses_4317
text_vectorization_inputO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	"
sequential_4304:	н-@%
sequential_4306:@@
sequential_4308:@!
sequential_4310:@
sequential_4312:
identityЂ"sequential/StatefulPartitionedCallЂ>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2
-text_vectorization/UnicodeSplit/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџВ
'text_vectorization/UnicodeSplit/ReshapeReshapetext_vectorization_input6text_vectorization/UnicodeSplit/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџП
-text_vectorization/UnicodeSplit/UnicodeDecodeUnicodeDecode0text_vectorization/UnicodeSplit/Reshape:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
input_encodingUTF-8
?text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
;text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims
ExpandDims;text_vectorization/UnicodeSplit/UnicodeDecode:char_values:0Htext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџж
Rtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ShapeShapeDtext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	Њ
`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
Ztext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_sliceStridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0itext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЦ
Ptext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mulMuletext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1:output:0etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_maskш
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0PackTtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:
Xtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : О
Stext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concatConcatV2etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0:output:0etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3:output:0atext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:П
Ttext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ReshapeReshapeDtext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims:output:0\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat:output:0*
T0*
Tshape0	*#
_output_shapes
:џџџџџџџџџЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask
Rtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/ShapeShape]text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0*
T0*
_output_shapes
:*
out_type0	Х
{text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ч
}text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ч
}text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
utext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSlicevtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskб
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RИ
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV2etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: з
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R з
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRangetext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

Tidx0	*#
_output_shapes
:џџџџџџџџџЙ
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMultext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Const:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ
Wtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncodeUnicodeEncode]text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*#
_output_shapes
:џџџџџџџџџ*
output_encodingUTF-8
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:џџџџџџџџџщ
&text_vectorization/string_lookup/EqualEqual`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:џџџџџџџџџ
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:џџџџџџџџџ
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџq
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"џџџџџџџџњ       С
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:0:text_vectorization/UnicodeSplit/UnicodeDecode:row_splits:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:џџџџџџџџџњ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITSф
"sequential/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0sequential_4304sequential_4306sequential_4308sequential_4310sequential_4312*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4019п
activation/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_3941r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЌ
NoOpNoOp#^sequential/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:џџџџџџџџџ
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

E
__inference__creator_4914
identity: ЂMutableHashTable~
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_106*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
Ч

О
+__inference_sequential_1_layer_call_fn_3965
text_vectorization_input
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:	н-@
	unknown_4:@@
	unknown_5:@
	unknown_6:@
	unknown_7:
identityЂStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCalltext_vectorization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
	*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_3944o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
#
_output_shapes
:џџџџџџџџџ
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ф
ї
)__inference_sequential_layer_call_fn_3704
embedding_input
unknown:	н-@
	unknown_0:@@
	unknown_1:@
	unknown_2:@
	unknown_3:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_3691o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
(
_output_shapes
:џџџџџџџџџњ
)
_user_specified_nameembedding_input

+
__inference__destroyer_4909
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Т	
№
?__inference_dense_layer_call_and_return_conditional_losses_3684

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
й
н
F__inference_sequential_1_layer_call_and_return_conditional_losses_4451

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	=
*sequential_embedding_embedding_lookup_4424:	н-@S
=sequential_conv1d_conv1d_expanddims_1_readvariableop_resource:@@?
1sequential_conv1d_biasadd_readvariableop_resource:@A
/sequential_dense_matmul_readvariableop_resource:@>
0sequential_dense_biasadd_readvariableop_resource:
identityЂ(sequential/conv1d/BiasAdd/ReadVariableOpЂ4sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOpЂ'sequential/dense/BiasAdd/ReadVariableOpЂ&sequential/dense/MatMul/ReadVariableOpЂ%sequential/embedding/embedding_lookupЂ>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2
-text_vectorization/UnicodeSplit/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ 
'text_vectorization/UnicodeSplit/ReshapeReshapeinputs6text_vectorization/UnicodeSplit/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџП
-text_vectorization/UnicodeSplit/UnicodeDecodeUnicodeDecode0text_vectorization/UnicodeSplit/Reshape:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
input_encodingUTF-8
?text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
;text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims
ExpandDims;text_vectorization/UnicodeSplit/UnicodeDecode:char_values:0Htext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџж
Rtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ShapeShapeDtext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	Њ
`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
Ztext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_sliceStridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0itext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЦ
Ptext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mulMuletext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1:output:0etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_maskш
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0PackTtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:
Xtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : О
Stext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concatConcatV2etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0:output:0etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3:output:0atext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:П
Ttext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ReshapeReshapeDtext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims:output:0\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat:output:0*
T0*
Tshape0	*#
_output_shapes
:џџџџџџџџџЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask
Rtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/ShapeShape]text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0*
T0*
_output_shapes
:*
out_type0	Х
{text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ч
}text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ч
}text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
utext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSlicevtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskб
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RИ
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV2etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: з
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R з
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRangetext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

Tidx0	*#
_output_shapes
:џџџџџџџџџЙ
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMultext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Const:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ
Wtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncodeUnicodeEncode]text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*#
_output_shapes
:џџџџџџџџџ*
output_encodingUTF-8
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:џџџџџџџџџщ
&text_vectorization/string_lookup/EqualEqual`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:џџџџџџџџџ
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:џџџџџџџџџ
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџq
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"џџџџџџџџњ       С
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:0:text_vectorization/UnicodeSplit/UnicodeDecode:row_splits:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:џџџџџџџџџњ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS
sequential/CastCast?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*

DstT0*

SrcT0	*(
_output_shapes
:џџџџџџџџџњx
sequential/embedding/CastCastsequential/Cast:y:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџњ
%sequential/embedding/embedding_lookupResourceGather*sequential_embedding_embedding_lookup_4424sequential/embedding/Cast:y:0*
Tindices0*=
_class3
1/loc:@sequential/embedding/embedding_lookup/4424*,
_output_shapes
:џџџџџџџџџњ@*
dtype0р
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*=
_class3
1/loc:@sequential/embedding/embedding_lookup/4424*,
_output_shapes
:џџџџџџџџџњ@Ќ
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@r
'sequential/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџй
#sequential/conv1d/Conv1D/ExpandDims
ExpandDims9sequential/embedding/embedding_lookup/Identity_1:output:00sequential/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@Ж
4sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=sequential_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0k
)sequential/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ж
%sequential/conv1d/Conv1D/ExpandDims_1
ExpandDims<sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:02sequential/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@у
sequential/conv1d/Conv1DConv2D,sequential/conv1d/Conv1D/ExpandDims:output:0.sequential/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
Ѕ
 sequential/conv1d/Conv1D/SqueezeSqueeze!sequential/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ
(sequential/conv1d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0И
sequential/conv1d/BiasAddBiasAdd)sequential/conv1d/Conv1D/Squeeze:output:00sequential/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@y
sequential/conv1d/ReluRelu"sequential/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@w
5sequential/global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Т
#sequential/global_max_pooling1d/MaxMax$sequential/conv1d/Relu:activations:0>sequential/global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Б
sequential/dense/MatMulMatMul,sequential/global_max_pooling1d/Max:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
activation/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџe
IdentityIdentityactivation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџф
NoOpNoOp)^sequential/conv1d/BiasAdd/ReadVariableOp5^sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : : : : : 2T
(sequential/conv1d/BiasAdd/ReadVariableOp(sequential/conv1d/BiasAdd/ReadVariableOp2l
4sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp4sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ј
ъ
D__inference_sequential_layer_call_and_return_conditional_losses_3831
embedding_input!
embedding_3816:	н-@!
conv1d_3819:@@
conv1d_3821:@

dense_3825:@

dense_3827:
identityЂconv1d/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂ!embedding/StatefulPartitionedCallъ
!embedding/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_3816*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_3647
conv1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_3819conv1d_3821*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_3667я
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_3627
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0
dense_3825
dense_3827*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_3684u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЋ
NoOpNoOp^conv1d/StatefulPartitionedCall^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:Y U
(
_output_shapes
:џџџџџџџџџњ
)
_user_specified_nameembedding_input
#
Њ
D__inference_sequential_layer_call_and_return_conditional_losses_4778

inputs	2
embedding_embedding_lookup_4752:	н-@H
2conv1d_conv1d_expanddims_1_readvariableop_resource:@@4
&conv1d_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:
identityЂconv1d/BiasAdd/ReadVariableOpЂ)conv1d/Conv1D/ExpandDims_1/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂembedding/embedding_lookupV
CastCastinputs*

DstT0*

SrcT0	*(
_output_shapes
:џџџџџџџџџњb
embedding/CastCastCast:y:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџњр
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_4752embedding/Cast:y:0*
Tindices0*2
_class(
&$loc:@embedding/embedding_lookup/4752*,
_output_shapes
:џџџџџџџџџњ@*
dtype0П
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/4752*,
_output_shapes
:џџџџџџџџџњ@
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџИ
conv1d/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@ 
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Е
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@Т
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@l
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ё
global_max_pooling1d/MaxMaxconv1d/Relu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense/MatMulMatMul!global_max_pooling1d/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџe
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџь
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:P L
(
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Ч
`
D__inference_activation_layer_call_and_return_conditional_losses_3941

inputs
identityL
SigmoidSigmoidinputs*
T0*'
_output_shapes
:џџџџџџџџџS
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
§
љ
F__inference_sequential_1_layer_call_and_return_conditional_losses_4129

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	"
sequential_4116:	н-@%
sequential_4118:@@
sequential_4120:@!
sequential_4122:@
sequential_4124:
identityЂ"sequential/StatefulPartitionedCallЂ>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2
-text_vectorization/UnicodeSplit/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ 
'text_vectorization/UnicodeSplit/ReshapeReshapeinputs6text_vectorization/UnicodeSplit/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџП
-text_vectorization/UnicodeSplit/UnicodeDecodeUnicodeDecode0text_vectorization/UnicodeSplit/Reshape:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
input_encodingUTF-8
?text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
;text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims
ExpandDims;text_vectorization/UnicodeSplit/UnicodeDecode:char_values:0Htext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџж
Rtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ShapeShapeDtext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	Њ
`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
Ztext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_sliceStridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0itext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЦ
Ptext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mulMuletext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1:output:0etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_maskш
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0PackTtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:
Xtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : О
Stext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concatConcatV2etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0:output:0etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3:output:0atext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:П
Ttext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ReshapeReshapeDtext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims:output:0\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat:output:0*
T0*
Tshape0	*#
_output_shapes
:џџџџџџџџџЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask
Rtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/ShapeShape]text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0*
T0*
_output_shapes
:*
out_type0	Х
{text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ч
}text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ч
}text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
utext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSlicevtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskб
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RИ
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV2etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: з
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R з
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRangetext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

Tidx0	*#
_output_shapes
:џџџџџџџџџЙ
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMultext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Const:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ
Wtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncodeUnicodeEncode]text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*#
_output_shapes
:џџџџџџџџџ*
output_encodingUTF-8
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:џџџџџџџџџщ
&text_vectorization/string_lookup/EqualEqual`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:џџџџџџџџџ
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:џџџџџџџџџ
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџq
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"џџџџџџџџњ       С
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:0:text_vectorization/UnicodeSplit/UnicodeDecode:row_splits:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:џџџџџџџџџњ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITSф
"sequential/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0sequential_4116sequential_4118sequential_4120sequential_4122sequential_4124*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4019п
activation/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_3941r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЌ
NoOpNoOp#^sequential/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

j
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_3627

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	

C__inference_embedding_layer_call_and_return_conditional_losses_3647

inputs(
embedding_lookup_3641:	н-@
identityЂembedding_lookupV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџњИ
embedding_lookupResourceGatherembedding_lookup_3641Cast:y:0*
Tindices0*(
_class
loc:@embedding_lookup/3641*,
_output_shapes
:џџџџџџџџџњ@*
dtype0Ё
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/3641*,
_output_shapes
:џџџџџџџџџњ@
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@x
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:џџџџџџџџџњ: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
й
н
F__inference_sequential_1_layer_call_and_return_conditional_losses_4539

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	=
*sequential_embedding_embedding_lookup_4512:	н-@S
=sequential_conv1d_conv1d_expanddims_1_readvariableop_resource:@@?
1sequential_conv1d_biasadd_readvariableop_resource:@A
/sequential_dense_matmul_readvariableop_resource:@>
0sequential_dense_biasadd_readvariableop_resource:
identityЂ(sequential/conv1d/BiasAdd/ReadVariableOpЂ4sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOpЂ'sequential/dense/BiasAdd/ReadVariableOpЂ&sequential/dense/MatMul/ReadVariableOpЂ%sequential/embedding/embedding_lookupЂ>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2
-text_vectorization/UnicodeSplit/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ 
'text_vectorization/UnicodeSplit/ReshapeReshapeinputs6text_vectorization/UnicodeSplit/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџП
-text_vectorization/UnicodeSplit/UnicodeDecodeUnicodeDecode0text_vectorization/UnicodeSplit/Reshape:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
input_encodingUTF-8
?text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
;text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims
ExpandDims;text_vectorization/UnicodeSplit/UnicodeDecode:char_values:0Htext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџж
Rtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ShapeShapeDtext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	Њ
`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
Ztext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_sliceStridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0itext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЦ
Ptext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mulMuletext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1:output:0etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_maskш
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0PackTtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:
Xtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : О
Stext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concatConcatV2etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0:output:0etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3:output:0atext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:П
Ttext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ReshapeReshapeDtext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims:output:0\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat:output:0*
T0*
Tshape0	*#
_output_shapes
:џџџџџџџџџЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask
Rtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/ShapeShape]text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0*
T0*
_output_shapes
:*
out_type0	Х
{text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ч
}text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ч
}text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
utext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSlicevtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskб
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RИ
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV2etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: з
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R з
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRangetext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

Tidx0	*#
_output_shapes
:џџџџџџџџџЙ
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMultext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Const:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ
Wtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncodeUnicodeEncode]text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*#
_output_shapes
:џџџџџџџџџ*
output_encodingUTF-8
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:џџџџџџџџџщ
&text_vectorization/string_lookup/EqualEqual`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:џџџџџџџџџ
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:џџџџџџџџџ
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџq
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"џџџџџџџџњ       С
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:0:text_vectorization/UnicodeSplit/UnicodeDecode:row_splits:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:џџџџџџџџџњ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITS
sequential/CastCast?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0*

DstT0*

SrcT0	*(
_output_shapes
:џџџџџџџџџњx
sequential/embedding/CastCastsequential/Cast:y:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџњ
%sequential/embedding/embedding_lookupResourceGather*sequential_embedding_embedding_lookup_4512sequential/embedding/Cast:y:0*
Tindices0*=
_class3
1/loc:@sequential/embedding/embedding_lookup/4512*,
_output_shapes
:џџџџџџџџџњ@*
dtype0р
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0*
T0*=
_class3
1/loc:@sequential/embedding/embedding_lookup/4512*,
_output_shapes
:џџџџџџџџџњ@Ќ
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@r
'sequential/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџй
#sequential/conv1d/Conv1D/ExpandDims
ExpandDims9sequential/embedding/embedding_lookup/Identity_1:output:00sequential/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@Ж
4sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp=sequential_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0k
)sequential/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ж
%sequential/conv1d/Conv1D/ExpandDims_1
ExpandDims<sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:02sequential/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@у
sequential/conv1d/Conv1DConv2D,sequential/conv1d/Conv1D/ExpandDims:output:0.sequential/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides
Ѕ
 sequential/conv1d/Conv1D/SqueezeSqueeze!sequential/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ
(sequential/conv1d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0И
sequential/conv1d/BiasAddBiasAdd)sequential/conv1d/Conv1D/Squeeze:output:00sequential/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@y
sequential/conv1d/ReluRelu"sequential/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@w
5sequential/global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Т
#sequential/global_max_pooling1d/MaxMax$sequential/conv1d/Relu:activations:0>sequential/global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Б
sequential/dense/MatMulMatMul,sequential/global_max_pooling1d/Max:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
activation/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџe
IdentityIdentityactivation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџф
NoOpNoOp)^sequential/conv1d/BiasAdd/ReadVariableOp5^sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : : : : : 2T
(sequential/conv1d/BiasAdd/ReadVariableOp(sequential/conv1d/BiasAdd/ReadVariableOp2l
4sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp4sequential/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
	

C__inference_embedding_layer_call_and_return_conditional_losses_4836

inputs(
embedding_lookup_4830:	н-@
identityЂembedding_lookupV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџњИ
embedding_lookupResourceGatherembedding_lookup_4830Cast:y:0*
Tindices0*(
_class
loc:@embedding_lookup/4830*,
_output_shapes
:џџџџџџџџџњ@*
dtype0Ё
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/4830*,
_output_shapes
:џџџџџџџџџњ@
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@x
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:џџџџџџџџџњ: 2$
embedding_lookupembedding_lookup:P L
(
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs

с
D__inference_sequential_layer_call_and_return_conditional_losses_3767

inputs!
embedding_3752:	н-@!
conv1d_3755:@@
conv1d_3757:@

dense_3761:@

dense_3763:
identityЂconv1d/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂ!embedding/StatefulPartitionedCallс
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_3752*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_3647
conv1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_3755conv1d_3757*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_3667я
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_3627
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0
dense_3761
dense_3763*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_3684u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЋ
NoOpNoOp^conv1d/StatefulPartitionedCall^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Ѓc
Ё
__inference_adapt_step_4621
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	ЂIteratorGetNextЂ(None_lookup_table_find/LookupTableFindV2Ђ,None_lookup_table_insert/LookupTableInsertV2Љ
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:џџџџџџџџџ*"
output_shapes
:џџџџџџџџџ*
output_types
2m
UnicodeSplit/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
UnicodeSplit/ReshapeReshapeIteratorGetNext:components:0#UnicodeSplit/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ
UnicodeSplit/UnicodeDecodeUnicodeDecodeUnicodeSplit/Reshape:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
input_encodingUTF-8n
,UnicodeSplit/RaggedExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Щ
(UnicodeSplit/RaggedExpandDims/ExpandDims
ExpandDims(UnicodeSplit/UnicodeDecode:char_values:05UnicodeSplit/RaggedExpandDims/ExpandDims/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџА
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ShapeShape1UnicodeSplit/RaggedExpandDims/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	
MUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѓ
GUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_sliceStridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0VUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ћ
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ћ
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask
=UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mulMulRUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1:output:0RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: 
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ѕ
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_maskТ
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0PackAUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:
EUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ђ
@UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concatConcatV2RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0:output:0RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3:output:0NUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:
AUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ReshapeReshape1UnicodeSplit/RaggedExpandDims/ExpandDims:output:0IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat:output:0*
T0*
Tshape0	*#
_output_shapes
:џџџџџџџџџ
OUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
QUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ћ
IUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4StridedSliceHUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0XUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1:output:0ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask
?UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rф
ZUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/ShapeShapeJUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0*
T0*
_output_shapes
:*
out_type0	В
hUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Д
jUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Д
jUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:њ
bUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSlicecUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0qUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0sUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0sUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskН
{UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rў
yUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV2RUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4:output:0UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: Ф
UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R Ф
UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 RЮ
{UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRangeUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0}UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

Tidx0	*#
_output_shapes
:џџџџџџџџџџ
yUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMulUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0HUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Const:output:0*
T0	*#
_output_shapes
:џџџџџџџџџс
DUnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncodeUnicodeEncodeJUnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0}UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*#
_output_shapes
:џџџџџџџџџ*
output_encodingUTF-8Ю
UniqueWithCountsUniqueWithCountsMUnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0*
T0*A
_output_shapes/
-:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
out_idx0	Ё
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 

E
)__inference_activation_layer_call_fn_4814

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_3941`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
э
и
__inference_restore_fn_4951
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identityЂ2MutableHashTable_table_restore/LookupTableImportV2
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1

)
__inference_<lambda>_4964
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Щ
ю
)__inference_sequential_layer_call_fn_4642

inputs
unknown:	н-@
	unknown_0:@@
	unknown_1:@
	unknown_2:@
	unknown_3:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_3691o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Щ
ю
)__inference_sequential_layer_call_fn_4657

inputs
unknown:	н-@
	unknown_0:@@
	unknown_1:@
	unknown_2:@
	unknown_3:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_3767o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Є
}
(__inference_embedding_layer_call_fn_4826

inputs
unknown:	н-@
identityЂStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_3647t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
:џџџџџџџџџњ: 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
ѕ
ю
__inference_<lambda>_49596
2key_value_init228_lookuptableimportv2_table_handle.
*key_value_init228_lookuptableimportv2_keys0
,key_value_init228_lookuptableimportv2_values	
identityЂ%key_value_init228/LookupTableImportV2ї
%key_value_init228/LookupTableImportV2LookupTableImportV22key_value_init228_lookuptableimportv2_table_handle*key_value_init228_lookuptableimportv2_keys,key_value_init228_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: n
NoOpNoOp&^key_value_init228/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :]:]2N
%key_value_init228/LookupTableImportV2%key_value_init228/LookupTableImportV2: 

_output_shapes
:]: 

_output_shapes
:]

j
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_4872

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
#
Њ
D__inference_sequential_layer_call_and_return_conditional_losses_4019

inputs	2
embedding_embedding_lookup_3993:	н-@H
2conv1d_conv1d_expanddims_1_readvariableop_resource:@@4
&conv1d_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:
identityЂconv1d/BiasAdd/ReadVariableOpЂ)conv1d/Conv1D/ExpandDims_1/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂembedding/embedding_lookupV
CastCastinputs*

DstT0*

SrcT0	*(
_output_shapes
:џџџџџџџџџњb
embedding/CastCastCast:y:0*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџњр
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_3993embedding/Cast:y:0*
Tindices0*2
_class(
&$loc:@embedding/embedding_lookup/3993*,
_output_shapes
:џџџџџџџџџњ@*
dtype0П
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/3993*,
_output_shapes
:џџџџџџџџџњ@
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџИ
conv1d/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@ 
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Е
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@Т
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@l
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ё
global_max_pooling1d/MaxMaxconv1d/Relu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense/MatMulMatMul!global_max_pooling1d/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџe
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџь
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:P L
(
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Г

F__inference_sequential_1_layer_call_and_return_conditional_losses_4245
text_vectorization_inputO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	"
sequential_4232:	н-@%
sequential_4234:@@
sequential_4236:@!
sequential_4238:@
sequential_4240:
identityЂ"sequential/StatefulPartitionedCallЂ>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2
-text_vectorization/UnicodeSplit/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџВ
'text_vectorization/UnicodeSplit/ReshapeReshapetext_vectorization_input6text_vectorization/UnicodeSplit/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџП
-text_vectorization/UnicodeSplit/UnicodeDecodeUnicodeDecode0text_vectorization/UnicodeSplit/Reshape:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
input_encodingUTF-8
?text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
;text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims
ExpandDims;text_vectorization/UnicodeSplit/UnicodeDecode:char_values:0Htext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџж
Rtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ShapeShapeDtext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	Њ
`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
Ztext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_sliceStridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0itext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЦ
Ptext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mulMuletext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1:output:0etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_maskш
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0PackTtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:
Xtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : О
Stext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concatConcatV2etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0:output:0etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3:output:0atext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:П
Ttext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ReshapeReshapeDtext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims:output:0\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat:output:0*
T0*
Tshape0	*#
_output_shapes
:џџџџџџџџџЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask
Rtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/ShapeShape]text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0*
T0*
_output_shapes
:*
out_type0	Х
{text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ч
}text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ч
}text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
utext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSlicevtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskб
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RИ
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV2etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: з
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R з
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRangetext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

Tidx0	*#
_output_shapes
:џџџџџџџџџЙ
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMultext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Const:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ
Wtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncodeUnicodeEncode]text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*#
_output_shapes
:џџџџџџџџџ*
output_encodingUTF-8
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:џџџџџџџџџщ
&text_vectorization/string_lookup/EqualEqual`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:џџџџџџџџџ
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:џџџџџџџџџ
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџq
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"џџџџџџџџњ       С
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:0:text_vectorization/UnicodeSplit/UnicodeDecode:row_splits:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:џџџџџџџџџњ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITSф
"sequential/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0sequential_4232sequential_4234sequential_4236sequential_4238sequential_4240*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_3924п
activation/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_3941r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЌ
NoOpNoOp#^sequential/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:] Y
#
_output_shapes
:џџџџџџџџџ
2
_user_specified_nametext_vectorization_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
в

%__inference_conv1d_layer_call_fn_4845

inputs
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_3667t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџњ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
Ј
ъ
D__inference_sequential_layer_call_and_return_conditional_losses_3813
embedding_input!
embedding_3798:	н-@!
conv1d_3801:@@
conv1d_3803:@

dense_3807:@

dense_3809:
identityЂconv1d/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂ!embedding/StatefulPartitionedCallъ
!embedding/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_3798*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_3647
conv1d/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0conv1d_3801conv1d_3803*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџњ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_3667я
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_3627
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0
dense_3807
dense_3809*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_3684u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЋ
NoOpNoOp^conv1d/StatefulPartitionedCall^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall:Y U
(
_output_shapes
:џџџџџџџџџњ
)
_user_specified_nameembedding_input
Њ"
Њ
D__inference_sequential_layer_call_and_return_conditional_losses_4717

inputs2
embedding_embedding_lookup_4691:	н-@H
2conv1d_conv1d_expanddims_1_readvariableop_resource:@@4
&conv1d_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:
identityЂconv1d/BiasAdd/ReadVariableOpЂ)conv1d/Conv1D/ExpandDims_1/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂembedding/embedding_lookup`
embedding/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџњр
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_4691embedding/Cast:y:0*
Tindices0*2
_class(
&$loc:@embedding/embedding_lookup/4691*,
_output_shapes
:џџџџџџџџџњ@*
dtype0П
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/4691*,
_output_shapes
:џџџџџџџџџњ@
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџИ
conv1d/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@ 
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Е
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@Т
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@l
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ё
global_max_pooling1d/MaxMaxconv1d/Relu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense/MatMulMatMul!global_max_pooling1d/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџe
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџь
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:P L
(
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
Њ"
Њ
D__inference_sequential_layer_call_and_return_conditional_losses_4747

inputs2
embedding_embedding_lookup_4721:	н-@H
2conv1d_conv1d_expanddims_1_readvariableop_resource:@@4
&conv1d_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:
identityЂconv1d/BiasAdd/ReadVariableOpЂ)conv1d/Conv1D/ExpandDims_1/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂembedding/embedding_lookup`
embedding/CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:џџџџџџџџџњр
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_4721embedding/Cast:y:0*
Tindices0*2
_class(
&$loc:@embedding/embedding_lookup/4721*,
_output_shapes
:џџџџџџџџџњ@*
dtype0П
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/4721*,
_output_shapes
:џџџџџџџџџњ@
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџИ
conv1d/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@ 
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Е
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@Т
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџ
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@l
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ё
global_max_pooling1d/MaxMaxconv1d/Relu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense/MatMulMatMul!global_max_pooling1d/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџe
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџь
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:P L
(
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
И

$__inference_dense_layer_call_fn_4881

inputs
unknown:@
	unknown_0:
identityЂStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_3684o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
зv

 __inference__traced_restore_5182
file_prefix8
%assignvariableop_embedding_embeddings:	н-@6
 assignvariableop_1_conv1d_kernel:@@,
assignvariableop_2_conv1d_bias:@1
assignvariableop_3_dense_kernel:@+
assignvariableop_4_dense_bias:M
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: &
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: #
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: %
assignvariableop_14_total_2: %
assignvariableop_15_count_2: %
assignvariableop_16_total_3: %
assignvariableop_17_count_3: B
/assignvariableop_18_adam_embedding_embeddings_m:	н-@>
(assignvariableop_19_adam_conv1d_kernel_m:@@4
&assignvariableop_20_adam_conv1d_bias_m:@9
'assignvariableop_21_adam_dense_kernel_m:@3
%assignvariableop_22_adam_dense_bias_m:B
/assignvariableop_23_adam_embedding_embeddings_v:	н-@>
(assignvariableop_24_adam_conv1d_kernel_v:@@4
&assignvariableop_25_adam_conv1d_bias_v:@9
'assignvariableop_26_adam_dense_kernel_v:@3
%assignvariableop_27_adam_dense_bias_v:
identity_29ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ2MutableHashTable_table_restore/LookupTableImportV2Г
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*й
valueЯBЬB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB>layer_with_weights-1/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-1/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEBIlayer_with_weights-1/keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/4/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/5/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWvariables/1/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/2/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/3/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/4/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWvariables/5/.OPTIMIZER_SLOT/layer_with_weights-1/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЎ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B К
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv1d_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:5RestoreV2:tensors:6*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 ]

Identity_5IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0^

Identity_8IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0^

Identity_9IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_2Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_3Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_3Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_18AssignVariableOp/assignvariableop_18_adam_embedding_embeddings_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_conv1d_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_conv1d_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_dense_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_23AssignVariableOp/assignvariableop_23_adam_embedding_embeddings_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv1d_kernel_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp&assignvariableop_25_adam_conv1d_bias_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_dense_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ь
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_29IdentityIdentity_28:output:0^NoOp_1*
T0*
_output_shapes
: й
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_29Identity_29:output:0*O
_input_shapes>
<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
Ы

@__inference_conv1d_layer_call_and_return_conditional_losses_4861

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџњ@*
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@*
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџњ@U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџњ@f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:џџџџџџџџџњ@
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџњ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџњ@
 
_user_specified_nameinputs
Ч
`
D__inference_activation_layer_call_and_return_conditional_losses_4819

inputs
identityL
SigmoidSigmoidinputs*
T0*'
_output_shapes
:џџџџџџџџџS
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ
ю
)__inference_sequential_layer_call_fn_4672

inputs	
unknown:	н-@
	unknown_0:@@
	unknown_1:@
	unknown_2:@
	unknown_3:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_3924o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџњ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџњ
 
_user_specified_nameinputs
§
љ
F__inference_sequential_1_layer_call_and_return_conditional_losses_3944

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	"
sequential_3925:	н-@%
sequential_3927:@@
sequential_3929:@!
sequential_3931:@
sequential_3933:
identityЂ"sequential/StatefulPartitionedCallЂ>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2
-text_vectorization/UnicodeSplit/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ 
'text_vectorization/UnicodeSplit/ReshapeReshapeinputs6text_vectorization/UnicodeSplit/Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџП
-text_vectorization/UnicodeSplit/UnicodeDecodeUnicodeDecode0text_vectorization/UnicodeSplit/Reshape:output:0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
input_encodingUTF-8
?text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
;text_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims
ExpandDims;text_vectorization/UnicodeSplit/UnicodeDecode:char_values:0Htext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџж
Rtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ShapeShapeDtext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	Њ
`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
Ztext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_sliceStridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0itext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_1:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЦ
Ptext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mulMuletext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_1:output:0etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: Ќ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_maskш
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0PackTtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:
Xtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : О
Stext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concatConcatV2etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/values_0:output:0etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_3:output:0atext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:П
Ttext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ReshapeReshapeDtext_vectorization/UnicodeSplit/RaggedExpandDims/ExpandDims:output:0\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/concat:output:0*
T0*
Tshape0	*#
_output_shapes
:џџџџџџџџџЌ
btext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ў
dtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
\text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4StridedSlice[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Shape:output:0ktext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_1:output:0mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask
Rtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
mtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/ShapeShape]text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0*
T0*
_output_shapes
:*
out_type0	Х
{text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ч
}text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ч
}text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
utext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSlicevtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskб
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RИ
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV2etext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/strided_slice_4:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: з
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R з
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRangetext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

Tidx0	*#
_output_shapes
:џџџџџџџџџЙ
text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMultext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0[text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Const:output:0*
T0	*#
_output_shapes
:џџџџџџџџџ
Wtext_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncodeUnicodeEncode]text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/Reshape:output:0text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*#
_output_shapes
:џџџџџџџџџ*
output_encodingUTF-8
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:џџџџџџџџџщ
&text_vectorization/string_lookup/EqualEqual`text_vectorization/UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode/UnicodeEncode:output:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:џџџџџџџџџ
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:џџџџџџџџџ
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:џџџџџџџџџq
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"џџџџџџџџњ       С
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:0:text_vectorization/UnicodeSplit/UnicodeDecode:row_splits:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:џџџџџџџџџњ*
num_row_partition_tensors*%
row_partition_types

ROW_SPLITSф
"sequential/StatefulPartitionedCallStatefulPartitionedCall?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0sequential_3925sequential_3927sequential_3929sequential_3931sequential_3933*
Tin

2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_3924п
activation/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_3941r
IdentityIdentity#activation/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЌ
NoOpNoOp#^sequential/StatefulPartitionedCall?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџ: : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
И
Ѓ
__inference_save_fn_4943
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	Ђ?MutableHashTable_lookup_table_export_values/LookupTableExportV2
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::K
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: O
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: 

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: 

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key"лL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Э
serving_defaultЙ
Y
text_vectorization_input=
*serving_default_text_vectorization_input:0џџџџџџџџџ@

activation2
StatefulPartitionedCall_1:0џџџџџџџџџtensorflow/serving/predict:ЮЂ
С
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
P
_lookup_layer
	keras_api
_adapt_function"
_tf_keras_layer
Л
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer
C
!1
"2
#3
$4
%5"
trackable_list_wrapper
C
!0
"1
#2
$3
%4"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
њ2ї
+__inference_sequential_1_layer_call_fn_3965
+__inference_sequential_1_layer_call_fn_4340
+__inference_sequential_1_layer_call_fn_4363
+__inference_sequential_1_layer_call_fn_4173Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
F__inference_sequential_1_layer_call_and_return_conditional_losses_4451
F__inference_sequential_1_layer_call_and_return_conditional_losses_4539
F__inference_sequential_1_layer_call_and_return_conditional_losses_4245
F__inference_sequential_1_layer_call_and_return_conditional_losses_4317Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
лBи
__inference__wrapped_model_3617text_vectorization_input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
,
+serving_default"
signature_map
L
,lookup_table
-token_counts
.	keras_api"
_tf_keras_layer
"
_generic_user_object
Н2К
__inference_adapt_step_4621
В
FullArgSpec
args

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Е
!
embeddings
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

"kernel
#bias
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

$kernel
%bias
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
З
Giter

Hbeta_1

Ibeta_2
	Jdecay
Klearning_rate!m"m#m$m%m!v"v#v$v%v"
	optimizer
C
!0
"1
#2
$3
%4"
trackable_list_wrapper
C
!0
"1
#2
$3
%4"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ш2Х
)__inference_sequential_layer_call_fn_3704
)__inference_sequential_layer_call_fn_4642
)__inference_sequential_layer_call_fn_4657
)__inference_sequential_layer_call_fn_3795
)__inference_sequential_layer_call_fn_4672
)__inference_sequential_layer_call_fn_4687Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ъ2ч
D__inference_sequential_layer_call_and_return_conditional_losses_4717
D__inference_sequential_layer_call_and_return_conditional_losses_4747
D__inference_sequential_layer_call_and_return_conditional_losses_3813
D__inference_sequential_layer_call_and_return_conditional_losses_3831
D__inference_sequential_layer_call_and_return_conditional_losses_4778
D__inference_sequential_layer_call_and_return_conditional_losses_4809Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
г2а
)__inference_activation_layer_call_fn_4814Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_activation_layer_call_and_return_conditional_losses_4819Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
':%	н-@2embedding/embeddings
#:!@@2conv1d/kernel
:@2conv1d/bias
:@2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
кBз
"__inference_signature_wrapper_4564text_vectorization_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
j
X_initializer
Y_create_resource
Z_initialize
[_destroy_resourceR jCustom.StaticHashTable
Q
\_create_resource
]_initialize
^_destroy_resourceR Z
table
"
_generic_user_object
'
!0"
trackable_list_wrapper
'
!0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
в2Я
(__inference_embedding_layer_call_fn_4826Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_embedding_layer_call_and_return_conditional_losses_4836Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
Я2Ь
%__inference_conv1d_layer_call_fn_4845Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъ2ч
@__inference_conv1d_layer_call_and_return_conditional_losses_4861Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
н2к
3__inference_global_max_pooling1d_layer_call_fn_4866Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ј2ѕ
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_4872Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
Ю2Ы
$__inference_dense_layer_call_fn_4881Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
щ2ц
?__inference_dense_layer_call_and_return_conditional_losses_4891Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	utotal
	vcount
w	variables
x	keras_api"
_tf_keras_metric
^
	ytotal
	zcount
{
_fn_kwargs
|	variables
}	keras_api"
_tf_keras_metric
"
_generic_user_object
А2­
__inference__creator_4896
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference__initializer_4904
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference__destroyer_4909
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
А2­
__inference__creator_4914
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
Д2Б
__inference__initializer_4919
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
В2Џ
__inference__destroyer_4924
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
P
	~total
	count
	variables
	keras_api"
_tf_keras_metric
c

total

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
u0
v1"
trackable_list_wrapper
-
w	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
y0
z1"
trackable_list_wrapper
-
|	variables"
_generic_user_object
:  (2total
:  (2count
.
~0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
,:*	н-@2Adam/embedding/embeddings/m
(:&@@2Adam/conv1d/kernel/m
:@2Adam/conv1d/bias/m
#:!@2Adam/dense/kernel/m
:2Adam/dense/bias/m
,:*	н-@2Adam/embedding/embeddings/v
(:&@@2Adam/conv1d/kernel/v
:@2Adam/conv1d/bias/v
#:!@2Adam/dense/kernel/v
:2Adam/dense/bias/v
мBй
__inference_save_fn_4943checkpoint_key"Њ
В
FullArgSpec
args
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ	
 
Bџ
__inference_restore_fn_4951restored_tensors_0restored_tensors_1"Е
В
FullArgSpec
args 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ
	
		
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_55
__inference__creator_4896Ђ

Ђ 
Њ " 5
__inference__creator_4914Ђ

Ђ 
Њ " 7
__inference__destroyer_4909Ђ

Ђ 
Њ " 7
__inference__destroyer_4924Ђ

Ђ 
Њ " @
__inference__initializer_4904,Ђ

Ђ 
Њ " 9
__inference__initializer_4919Ђ

Ђ 
Њ " Њ
__inference__wrapped_model_3617,!"#$%=Ђ:
3Ђ0
.+
text_vectorization_inputџџџџџџџџџ
Њ "7Њ4
2

activation$!

activationџџџџџџџџџ 
D__inference_activation_layer_call_and_return_conditional_losses_4819X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 x
)__inference_activation_layer_call_fn_4814K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџi
__inference_adapt_step_4621J-?Ђ<
5Ђ2
0-Ђ
џџџџџџџџџIteratorSpec 
Њ "
 Њ
@__inference_conv1d_layer_call_and_return_conditional_losses_4861f"#4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ@
Њ "*Ђ'
 
0џџџџџџџџџњ@
 
%__inference_conv1d_layer_call_fn_4845Y"#4Ђ1
*Ђ'
%"
inputsџџџџџџџџџњ@
Њ "џџџџџџџџџњ@
?__inference_dense_layer_call_and_return_conditional_losses_4891\$%/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ
 w
$__inference_dense_layer_call_fn_4881O$%/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџЈ
C__inference_embedding_layer_call_and_return_conditional_losses_4836a!0Ђ-
&Ђ#
!
inputsџџџџџџџџџњ
Њ "*Ђ'
 
0џџџџџџџџџњ@
 
(__inference_embedding_layer_call_fn_4826T!0Ђ-
&Ђ#
!
inputsџџџџџџџџџњ
Њ "џџџџџџџџџњ@Щ
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_4872wEЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 Ё
3__inference_global_max_pooling1d_layer_call_fn_4866jEЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "!џџџџџџџџџџџџџџџџџџx
__inference_restore_fn_4951Y-KЂH
AЂ>

restored_tensors_0

restored_tensors_1	
Њ " 
__inference_save_fn_4943і-&Ђ#
Ђ

checkpoint_key 
Њ "ШФ
`Њ]

name
0/name 
#

slice_spec
0/slice_spec 

tensor
0/tensor
`Њ]

name
1/name 
#

slice_spec
1/slice_spec 

tensor
1/tensor	Ц
F__inference_sequential_1_layer_call_and_return_conditional_losses_4245|,!"#$%EЂB
;Ђ8
.+
text_vectorization_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Ц
F__inference_sequential_1_layer_call_and_return_conditional_losses_4317|,!"#$%EЂB
;Ђ8
.+
text_vectorization_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Д
F__inference_sequential_1_layer_call_and_return_conditional_losses_4451j,!"#$%3Ђ0
)Ђ&

inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Д
F__inference_sequential_1_layer_call_and_return_conditional_losses_4539j,!"#$%3Ђ0
)Ђ&

inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
+__inference_sequential_1_layer_call_fn_3965o,!"#$%EЂB
;Ђ8
.+
text_vectorization_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
+__inference_sequential_1_layer_call_fn_4173o,!"#$%EЂB
;Ђ8
.+
text_vectorization_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
+__inference_sequential_1_layer_call_fn_4340],!"#$%3Ђ0
)Ђ&

inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
+__inference_sequential_1_layer_call_fn_4363],!"#$%3Ђ0
)Ђ&

inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџЙ
D__inference_sequential_layer_call_and_return_conditional_losses_3813q!"#$%AЂ>
7Ђ4
*'
embedding_inputџџџџџџџџџњ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Й
D__inference_sequential_layer_call_and_return_conditional_losses_3831q!"#$%AЂ>
7Ђ4
*'
embedding_inputџџџџџџџџџњ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 А
D__inference_sequential_layer_call_and_return_conditional_losses_4717h!"#$%8Ђ5
.Ђ+
!
inputsџџџџџџџџџњ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 А
D__inference_sequential_layer_call_and_return_conditional_losses_4747h!"#$%8Ђ5
.Ђ+
!
inputsџџџџџџџџџњ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 А
D__inference_sequential_layer_call_and_return_conditional_losses_4778h!"#$%8Ђ5
.Ђ+
!
inputsџџџџџџџџџњ	
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 А
D__inference_sequential_layer_call_and_return_conditional_losses_4809h!"#$%8Ђ5
.Ђ+
!
inputsџџџџџџџџџњ	
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
)__inference_sequential_layer_call_fn_3704d!"#$%AЂ>
7Ђ4
*'
embedding_inputџџџџџџџџџњ
p 

 
Њ "џџџџџџџџџ
)__inference_sequential_layer_call_fn_3795d!"#$%AЂ>
7Ђ4
*'
embedding_inputџџџџџџџџџњ
p

 
Њ "џџџџџџџџџ
)__inference_sequential_layer_call_fn_4642[!"#$%8Ђ5
.Ђ+
!
inputsџџџџџџџџџњ
p 

 
Њ "џџџџџџџџџ
)__inference_sequential_layer_call_fn_4657[!"#$%8Ђ5
.Ђ+
!
inputsџџџџџџџџџњ
p

 
Њ "џџџџџџџџџ
)__inference_sequential_layer_call_fn_4672[!"#$%8Ђ5
.Ђ+
!
inputsџџџџџџџџџњ	
p 

 
Њ "џџџџџџџџџ
)__inference_sequential_layer_call_fn_4687[!"#$%8Ђ5
.Ђ+
!
inputsџџџџџџџџџњ	
p

 
Њ "џџџџџџџџџЩ
"__inference_signature_wrapper_4564Ђ,!"#$%YЂV
Ђ 
OЊL
J
text_vectorization_input.+
text_vectorization_inputџџџџџџџџџ"7Њ4
2

activation$!

activationџџџџџџџџџ