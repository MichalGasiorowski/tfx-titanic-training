??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	??
?
AsString

input"T

output"
Ttype:
2		
"
	precisionint?????????"

scientificbool( "
shortestbool( "
widthint?????????"
fillstring 
?
BoostedTreesBucketize
float_values*num_features#
bucket_boundaries*num_features
buckets*num_features"
num_featuresint(
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
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
?
InitializeTableFromTextFileV2
table_handle
filename"
	key_indexint(0?????????"
value_indexint(0?????????"+

vocab_sizeint?????????(0?????????"
	delimiterstring	?
.
IsFinite
x"T
y
"
Ttype:
2
+
IsNan
x"T
y
"
Ttype:
2
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
2
LookupTableSizeV2
table_handle
size	?
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
n
NotEqual
x"T
y"T
z
""
Ttype:
2	
"$
incompatible_shape_errorbool(?
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
l
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

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
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
?
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
-
Sqrt
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
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
G
StringToHashBucketFast	
input

output	"
num_bucketsint(0
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"	transform*2.3.02v2.3.0-rc2-23-gb36436b087ԡ
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *)$?A
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *??SC
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *???A
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *"v E
I
Const_4Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
Const_5Const*
_output_shapes
: *
dtype0*?
value?B? B?/home/michal/artifact-store/tfx-titanic-training/20210425_225924/Transform/transform_graph/5/.temp_path/tftransform_tmp/vocab_compute_and_apply_vocabulary_vocabulary
I
Const_6Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
Const_7Const*
_output_shapes
: *
dtype0*?
value?B? B?/home/michal/artifact-store/tfx-titanic-training/20210425_225924/Transform/transform_graph/5/.temp_path/tftransform_tmp/vocab_compute_and_apply_vocabulary_1_vocabulary
I
Const_8Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
Const_9Const*
_output_shapes
: *
dtype0*?
value?B? B?/home/michal/artifact-store/tfx-titanic-training/20210425_225924/Transform/transform_graph/5/.temp_path/tftransform_tmp/vocab_compute_and_apply_vocabulary_2_vocabulary
M
Const_10Const*
_output_shapes
: *
dtype0*
valueB
 *($?A
M
Const_11Const*
_output_shapes
: *
dtype0*
valueB
 *?N)C
M
Const_12Const*
_output_shapes
: *
dtype0*
valueB
 *???A
M
Const_13Const*
_output_shapes
: *
dtype0*
valueB
 *"v E
?
%transform/inputs/Embarked/PlaceholderPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
?
'transform/inputs/Embarked/Placeholder_1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
p
'transform/inputs/Embarked/Placeholder_2Placeholder*
_output_shapes
:*
dtype0	*
shape:
?
#transform/inputs/Ticket/PlaceholderPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
?
%transform/inputs/Ticket/Placeholder_1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
n
%transform/inputs/Ticket/Placeholder_2Placeholder*
_output_shapes
:*
dtype0	*
shape:
?
 transform/inputs/Sex/PlaceholderPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
}
"transform/inputs/Sex/Placeholder_1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
k
"transform/inputs/Sex/Placeholder_2Placeholder*
_output_shapes
:*
dtype0	*
shape:
?
!transform/inputs/Name/PlaceholderPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
~
#transform/inputs/Name/Placeholder_1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
l
#transform/inputs/Name/Placeholder_2Placeholder*
_output_shapes
:*
dtype0	*
shape:
?
"transform/inputs/Cabin/PlaceholderPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????

$transform/inputs/Cabin/Placeholder_1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
m
$transform/inputs/Cabin/Placeholder_2Placeholder*
_output_shapes
:*
dtype0	*
shape:
?
 transform/inputs/Age/PlaceholderPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
}
"transform/inputs/Age/Placeholder_1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
k
"transform/inputs/Age/Placeholder_2Placeholder*
_output_shapes
:*
dtype0	*
shape:
?
!transform/inputs/Fare/PlaceholderPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
~
#transform/inputs/Fare/Placeholder_1Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
l
#transform/inputs/Fare/Placeholder_2Placeholder*
_output_shapes
:*
dtype0	*
shape:
?
"transform/inputs/Parch/PlaceholderPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????

$transform/inputs/Parch/Placeholder_1Placeholder*#
_output_shapes
:?????????*
dtype0	*
shape:?????????
m
$transform/inputs/Parch/Placeholder_2Placeholder*
_output_shapes
:*
dtype0	*
shape:
?
(transform/inputs/PassengerId/PlaceholderPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
?
*transform/inputs/PassengerId/Placeholder_1Placeholder*#
_output_shapes
:?????????*
dtype0	*
shape:?????????
s
*transform/inputs/PassengerId/Placeholder_2Placeholder*
_output_shapes
:*
dtype0	*
shape:
?
#transform/inputs/Pclass/PlaceholderPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
?
%transform/inputs/Pclass/Placeholder_1Placeholder*#
_output_shapes
:?????????*
dtype0	*
shape:?????????
n
%transform/inputs/Pclass/Placeholder_2Placeholder*
_output_shapes
:*
dtype0	*
shape:
?
"transform/inputs/SibSp/PlaceholderPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????

$transform/inputs/SibSp/Placeholder_1Placeholder*#
_output_shapes
:?????????*
dtype0	*
shape:?????????
m
$transform/inputs/SibSp/Placeholder_2Placeholder*
_output_shapes
:*
dtype0	*
shape:
?
%transform/inputs/Survived/PlaceholderPlaceholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
?
'transform/inputs/Survived/Placeholder_1Placeholder*#
_output_shapes
:?????????*
dtype0	*
shape:?????????
p
'transform/inputs/Survived/Placeholder_2Placeholder*
_output_shapes
:*
dtype0	*
shape:
?
1transform/inputs/inputs/Embarked/Placeholder_copyIdentity%transform/inputs/Embarked/Placeholder*
T0	*'
_output_shapes
:?????????
?
3transform/inputs/inputs/Embarked/Placeholder_1_copyIdentity'transform/inputs/Embarked/Placeholder_1*
T0*#
_output_shapes
:?????????
?
3transform/inputs/inputs/Embarked/Placeholder_2_copyIdentity'transform/inputs/Embarked/Placeholder_2*
T0	*
_output_shapes
:
?
/transform/inputs/inputs/Ticket/Placeholder_copyIdentity#transform/inputs/Ticket/Placeholder*
T0	*'
_output_shapes
:?????????
?
1transform/inputs/inputs/Ticket/Placeholder_1_copyIdentity%transform/inputs/Ticket/Placeholder_1*
T0*#
_output_shapes
:?????????
?
1transform/inputs/inputs/Ticket/Placeholder_2_copyIdentity%transform/inputs/Ticket/Placeholder_2*
T0	*
_output_shapes
:
?
,transform/inputs/inputs/Sex/Placeholder_copyIdentity transform/inputs/Sex/Placeholder*
T0	*'
_output_shapes
:?????????
?
.transform/inputs/inputs/Sex/Placeholder_1_copyIdentity"transform/inputs/Sex/Placeholder_1*
T0*#
_output_shapes
:?????????
?
.transform/inputs/inputs/Sex/Placeholder_2_copyIdentity"transform/inputs/Sex/Placeholder_2*
T0	*
_output_shapes
:
?
-transform/inputs/inputs/Name/Placeholder_copyIdentity!transform/inputs/Name/Placeholder*
T0	*'
_output_shapes
:?????????
?
/transform/inputs/inputs/Name/Placeholder_1_copyIdentity#transform/inputs/Name/Placeholder_1*
T0*#
_output_shapes
:?????????
?
/transform/inputs/inputs/Name/Placeholder_2_copyIdentity#transform/inputs/Name/Placeholder_2*
T0	*
_output_shapes
:
?
.transform/inputs/inputs/Cabin/Placeholder_copyIdentity"transform/inputs/Cabin/Placeholder*
T0	*'
_output_shapes
:?????????
?
0transform/inputs/inputs/Cabin/Placeholder_1_copyIdentity$transform/inputs/Cabin/Placeholder_1*
T0*#
_output_shapes
:?????????
?
0transform/inputs/inputs/Cabin/Placeholder_2_copyIdentity$transform/inputs/Cabin/Placeholder_2*
T0	*
_output_shapes
:
?
,transform/inputs/inputs/Age/Placeholder_copyIdentity transform/inputs/Age/Placeholder*
T0	*'
_output_shapes
:?????????
?
.transform/inputs/inputs/Age/Placeholder_1_copyIdentity"transform/inputs/Age/Placeholder_1*
T0*#
_output_shapes
:?????????
?
.transform/inputs/inputs/Age/Placeholder_2_copyIdentity"transform/inputs/Age/Placeholder_2*
T0	*
_output_shapes
:
?
-transform/inputs/inputs/Fare/Placeholder_copyIdentity!transform/inputs/Fare/Placeholder*
T0	*'
_output_shapes
:?????????
?
/transform/inputs/inputs/Fare/Placeholder_1_copyIdentity#transform/inputs/Fare/Placeholder_1*
T0*#
_output_shapes
:?????????
?
/transform/inputs/inputs/Fare/Placeholder_2_copyIdentity#transform/inputs/Fare/Placeholder_2*
T0	*
_output_shapes
:
?
.transform/inputs/inputs/Parch/Placeholder_copyIdentity"transform/inputs/Parch/Placeholder*
T0	*'
_output_shapes
:?????????
?
0transform/inputs/inputs/Parch/Placeholder_1_copyIdentity$transform/inputs/Parch/Placeholder_1*
T0	*#
_output_shapes
:?????????
?
0transform/inputs/inputs/Parch/Placeholder_2_copyIdentity$transform/inputs/Parch/Placeholder_2*
T0	*
_output_shapes
:
?
4transform/inputs/inputs/PassengerId/Placeholder_copyIdentity(transform/inputs/PassengerId/Placeholder*
T0	*'
_output_shapes
:?????????
?
6transform/inputs/inputs/PassengerId/Placeholder_1_copyIdentity*transform/inputs/PassengerId/Placeholder_1*
T0	*#
_output_shapes
:?????????
?
6transform/inputs/inputs/PassengerId/Placeholder_2_copyIdentity*transform/inputs/PassengerId/Placeholder_2*
T0	*
_output_shapes
:
?
/transform/inputs/inputs/Pclass/Placeholder_copyIdentity#transform/inputs/Pclass/Placeholder*
T0	*'
_output_shapes
:?????????
?
1transform/inputs/inputs/Pclass/Placeholder_1_copyIdentity%transform/inputs/Pclass/Placeholder_1*
T0	*#
_output_shapes
:?????????
?
1transform/inputs/inputs/Pclass/Placeholder_2_copyIdentity%transform/inputs/Pclass/Placeholder_2*
T0	*
_output_shapes
:
?
.transform/inputs/inputs/SibSp/Placeholder_copyIdentity"transform/inputs/SibSp/Placeholder*
T0	*'
_output_shapes
:?????????
?
0transform/inputs/inputs/SibSp/Placeholder_1_copyIdentity$transform/inputs/SibSp/Placeholder_1*
T0	*#
_output_shapes
:?????????
?
0transform/inputs/inputs/SibSp/Placeholder_2_copyIdentity$transform/inputs/SibSp/Placeholder_2*
T0	*
_output_shapes
:
?
1transform/inputs/inputs/Survived/Placeholder_copyIdentity%transform/inputs/Survived/Placeholder*
T0	*'
_output_shapes
:?????????
?
3transform/inputs/inputs/Survived/Placeholder_1_copyIdentity'transform/inputs/Survived/Placeholder_1*
T0	*#
_output_shapes
:?????????
?
3transform/inputs/inputs/Survived/Placeholder_2_copyIdentity'transform/inputs/Survived/Placeholder_2*
T0	*
_output_shapes
:
|
transform/IsFiniteIsFinite.transform/inputs/inputs/Age/Placeholder_1_copy*
T0*#
_output_shapes
:?????????
?
transform/boolean_mask/ShapeShape.transform/inputs/inputs/Age/Placeholder_1_copy*
T0*
_output_shapes
:*
out_type0
t
*transform/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
v
,transform/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
v
,transform/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
$transform/boolean_mask/strided_sliceStridedSlicetransform/boolean_mask/Shape*transform/boolean_mask/strided_slice/stack,transform/boolean_mask/strided_slice/stack_1,transform/boolean_mask/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask 
w
-transform/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
?
transform/boolean_mask/ProdProd$transform/boolean_mask/strided_slice-transform/boolean_mask/Prod/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
transform/boolean_mask/Shape_1Shape.transform/inputs/inputs/Age/Placeholder_1_copy*
T0*
_output_shapes
:*
out_type0
v
,transform/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
x
.transform/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
x
.transform/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
&transform/boolean_mask/strided_slice_1StridedSlicetransform/boolean_mask/Shape_1,transform/boolean_mask/strided_slice_1/stack.transform/boolean_mask/strided_slice_1/stack_1.transform/boolean_mask/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask*
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask 
?
transform/boolean_mask/Shape_2Shape.transform/inputs/inputs/Age/Placeholder_1_copy*
T0*
_output_shapes
:*
out_type0
v
,transform/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
x
.transform/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
x
.transform/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
&transform/boolean_mask/strided_slice_2StridedSlicetransform/boolean_mask/Shape_2,transform/boolean_mask/strided_slice_2/stack.transform/boolean_mask/strided_slice_2/stack_1.transform/boolean_mask/strided_slice_2/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
?
&transform/boolean_mask/concat/values_1Packtransform/boolean_mask/Prod*
N*
T0*
_output_shapes
:*

axis 
d
"transform/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
transform/boolean_mask/concatConcatV2&transform/boolean_mask/strided_slice_1&transform/boolean_mask/concat/values_1&transform/boolean_mask/strided_slice_2"transform/boolean_mask/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
?
transform/boolean_mask/ReshapeReshape.transform/inputs/inputs/Age/Placeholder_1_copytransform/boolean_mask/concat*
T0*
Tshape0*#
_output_shapes
:?????????
y
&transform/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
 transform/boolean_mask/Reshape_1Reshapetransform/IsFinite&transform/boolean_mask/Reshape_1/shape*
T0
*
Tshape0*#
_output_shapes
:?????????
y
transform/boolean_mask/WhereWhere transform/boolean_mask/Reshape_1*
T0
*'
_output_shapes
:?????????
?
transform/boolean_mask/SqueezeSqueezetransform/boolean_mask/Where*
T0	*#
_output_shapes
:?????????*
squeeze_dims

f
$transform/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
transform/boolean_mask/GatherV2GatherV2transform/boolean_mask/Reshapetransform/boolean_mask/Squeeze$transform/boolean_mask/GatherV2/axis*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????*

batch_dims 
z
 transform/mean/mean_and_var/SizeSizetransform/boolean_mask/GatherV2*
T0*
_output_shapes
: *
out_type0
?
 transform/mean/mean_and_var/CastCast transform/mean/mean_and_var/Size*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 
k
!transform/mean/mean_and_var/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
transform/mean/mean_and_var/SumSumtransform/boolean_mask/GatherV2!transform/mean/mean_and_var/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
#transform/mean/mean_and_var/truedivRealDivtransform/mean/mean_and_var/Sum transform/mean/mean_and_var/Cast*
T0*
_output_shapes
: 
?
transform/mean/mean_and_var/subSubtransform/boolean_mask/GatherV2#transform/mean/mean_and_var/truediv*
T0*#
_output_shapes
:?????????
{
"transform/mean/mean_and_var/SquareSquaretransform/mean/mean_and_var/sub*
T0*#
_output_shapes
:?????????
m
#transform/mean/mean_and_var/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
!transform/mean/mean_and_var/Sum_1Sum"transform/mean/mean_and_var/Square#transform/mean/mean_and_var/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
%transform/mean/mean_and_var/truediv_1RealDiv!transform/mean/mean_and_var/Sum_1 transform/mean/mean_and_var/Cast*
T0*
_output_shapes
: 
f
!transform/mean/mean_and_var/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    
h
'transform/mean/mean_and_var/PlaceholderPlaceholder*
_output_shapes
: *
dtype0*
shape: 
j
)transform/mean/mean_and_var/Placeholder_1Placeholder*
_output_shapes
: *
dtype0*
shape: 
g
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
i
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
i
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
transform/strided_sliceStridedSlice.transform/inputs/inputs/Age/Placeholder_2_copytransform/strided_slice/stacktransform/strided_slice/stack_1transform/strided_slice/stack_2*
Index0*
T0	*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
f
$transform/SparseTensor/dense_shape/1Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
"transform/SparseTensor/dense_shapePacktransform/strided_slice$transform/SparseTensor/dense_shape/1*
N*
T0	*
_output_shapes
:*

axis 
?
transform/SparseToDenseSparseToDense,transform/inputs/inputs/Age/Placeholder_copy"transform/SparseTensor/dense_shape.transform/inputs/inputs/Age/Placeholder_1_copyConst*
T0*
Tindices0	*'
_output_shapes
:?????????*
validate_indices(
c
transform/IsNanIsNantransform/SparseToDense*
T0*'
_output_shapes
:?????????
?
transform/SelectV2SelectV2transform/IsNanConsttransform/SparseToDense*
T0*'
_output_shapes
:?????????
u
transform/SqueezeSqueezetransform/SelectV2*
T0*#
_output_shapes
:?????????*
squeeze_dims

x
,transform/scale_to_z_score/mean_and_var/SizeSizetransform/Squeeze*
T0*
_output_shapes
: *
out_type0
?
,transform/scale_to_z_score/mean_and_var/CastCast,transform/scale_to_z_score/mean_and_var/Size*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 
w
-transform/scale_to_z_score/mean_and_var/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
+transform/scale_to_z_score/mean_and_var/SumSumtransform/Squeeze-transform/scale_to_z_score/mean_and_var/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
/transform/scale_to_z_score/mean_and_var/truedivRealDiv+transform/scale_to_z_score/mean_and_var/Sum,transform/scale_to_z_score/mean_and_var/Cast*
T0*
_output_shapes
: 
?
+transform/scale_to_z_score/mean_and_var/subSubtransform/Squeeze/transform/scale_to_z_score/mean_and_var/truediv*
T0*#
_output_shapes
:?????????
?
.transform/scale_to_z_score/mean_and_var/SquareSquare+transform/scale_to_z_score/mean_and_var/sub*
T0*#
_output_shapes
:?????????
y
/transform/scale_to_z_score/mean_and_var/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
-transform/scale_to_z_score/mean_and_var/Sum_1Sum.transform/scale_to_z_score/mean_and_var/Square/transform/scale_to_z_score/mean_and_var/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
1transform/scale_to_z_score/mean_and_var/truediv_1RealDiv-transform/scale_to_z_score/mean_and_var/Sum_1,transform/scale_to_z_score/mean_and_var/Cast*
T0*
_output_shapes
: 
r
-transform/scale_to_z_score/mean_and_var/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    
t
3transform/scale_to_z_score/mean_and_var/PlaceholderPlaceholder*
_output_shapes
: *
dtype0*
shape: 
v
5transform/scale_to_z_score/mean_and_var/Placeholder_1Placeholder*
_output_shapes
: *
dtype0*
shape: 
p
transform/scale_to_z_score/subSubtransform/SqueezeConst_10*
T0*#
_output_shapes
:?????????
R
transform/scale_to_z_score/SqrtSqrtConst_11*
T0*
_output_shapes
: 
j
%transform/scale_to_z_score/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
?
#transform/scale_to_z_score/NotEqualNotEqualtransform/scale_to_z_score/Sqrt%transform/scale_to_z_score/NotEqual/y*
T0*
_output_shapes
: *
incompatible_shape_error(
?
%transform/scale_to_z_score/zeros_like	ZerosLiketransform/scale_to_z_score/sub*
T0*#
_output_shapes
:?????????
?
transform/scale_to_z_score/CastCast#transform/scale_to_z_score/NotEqual*

DstT0*

SrcT0
*
Truncate( *
_output_shapes
: 
?
transform/scale_to_z_score/addAddV2%transform/scale_to_z_score/zeros_liketransform/scale_to_z_score/Cast*
T0*#
_output_shapes
:?????????
?
!transform/scale_to_z_score/Cast_1Casttransform/scale_to_z_score/add*

DstT0
*

SrcT0*
Truncate( *#
_output_shapes
:?????????
?
"transform/scale_to_z_score/truedivRealDivtransform/scale_to_z_score/subtransform/scale_to_z_score/Sqrt*
T0*#
_output_shapes
:?????????
?
#transform/scale_to_z_score/SelectV2SelectV2!transform/scale_to_z_score/Cast_1"transform/scale_to_z_score/truedivtransform/scale_to_z_score/sub*
T0*#
_output_shapes
:?????????

transform/IsFinite_1IsFinite/transform/inputs/inputs/Fare/Placeholder_1_copy*
T0*#
_output_shapes
:?????????
?
transform/boolean_mask_1/ShapeShape/transform/inputs/inputs/Fare/Placeholder_1_copy*
T0*
_output_shapes
:*
out_type0
v
,transform/boolean_mask_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
x
.transform/boolean_mask_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
x
.transform/boolean_mask_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
&transform/boolean_mask_1/strided_sliceStridedSlicetransform/boolean_mask_1/Shape,transform/boolean_mask_1/strided_slice/stack.transform/boolean_mask_1/strided_slice/stack_1.transform/boolean_mask_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask 
y
/transform/boolean_mask_1/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
?
transform/boolean_mask_1/ProdProd&transform/boolean_mask_1/strided_slice/transform/boolean_mask_1/Prod/reduction_indices*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
 transform/boolean_mask_1/Shape_1Shape/transform/inputs/inputs/Fare/Placeholder_1_copy*
T0*
_output_shapes
:*
out_type0
x
.transform/boolean_mask_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
z
0transform/boolean_mask_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
z
0transform/boolean_mask_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
(transform/boolean_mask_1/strided_slice_1StridedSlice transform/boolean_mask_1/Shape_1.transform/boolean_mask_1/strided_slice_1/stack0transform/boolean_mask_1/strided_slice_1/stack_10transform/boolean_mask_1/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask*
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask 
?
 transform/boolean_mask_1/Shape_2Shape/transform/inputs/inputs/Fare/Placeholder_1_copy*
T0*
_output_shapes
:*
out_type0
x
.transform/boolean_mask_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
z
0transform/boolean_mask_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
z
0transform/boolean_mask_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
(transform/boolean_mask_1/strided_slice_2StridedSlice transform/boolean_mask_1/Shape_2.transform/boolean_mask_1/strided_slice_2/stack0transform/boolean_mask_1/strided_slice_2/stack_10transform/boolean_mask_1/strided_slice_2/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
?
(transform/boolean_mask_1/concat/values_1Packtransform/boolean_mask_1/Prod*
N*
T0*
_output_shapes
:*

axis 
f
$transform/boolean_mask_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
transform/boolean_mask_1/concatConcatV2(transform/boolean_mask_1/strided_slice_1(transform/boolean_mask_1/concat/values_1(transform/boolean_mask_1/strided_slice_2$transform/boolean_mask_1/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
?
 transform/boolean_mask_1/ReshapeReshape/transform/inputs/inputs/Fare/Placeholder_1_copytransform/boolean_mask_1/concat*
T0*
Tshape0*#
_output_shapes
:?????????
{
(transform/boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
"transform/boolean_mask_1/Reshape_1Reshapetransform/IsFinite_1(transform/boolean_mask_1/Reshape_1/shape*
T0
*
Tshape0*#
_output_shapes
:?????????
}
transform/boolean_mask_1/WhereWhere"transform/boolean_mask_1/Reshape_1*
T0
*'
_output_shapes
:?????????
?
 transform/boolean_mask_1/SqueezeSqueezetransform/boolean_mask_1/Where*
T0	*#
_output_shapes
:?????????*
squeeze_dims

h
&transform/boolean_mask_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
!transform/boolean_mask_1/GatherV2GatherV2 transform/boolean_mask_1/Reshape transform/boolean_mask_1/Squeeze&transform/boolean_mask_1/GatherV2/axis*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????*

batch_dims 
~
"transform/mean_1/mean_and_var/SizeSize!transform/boolean_mask_1/GatherV2*
T0*
_output_shapes
: *
out_type0
?
"transform/mean_1/mean_and_var/CastCast"transform/mean_1/mean_and_var/Size*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 
m
#transform/mean_1/mean_and_var/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
!transform/mean_1/mean_and_var/SumSum!transform/boolean_mask_1/GatherV2#transform/mean_1/mean_and_var/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
%transform/mean_1/mean_and_var/truedivRealDiv!transform/mean_1/mean_and_var/Sum"transform/mean_1/mean_and_var/Cast*
T0*
_output_shapes
: 
?
!transform/mean_1/mean_and_var/subSub!transform/boolean_mask_1/GatherV2%transform/mean_1/mean_and_var/truediv*
T0*#
_output_shapes
:?????????

$transform/mean_1/mean_and_var/SquareSquare!transform/mean_1/mean_and_var/sub*
T0*#
_output_shapes
:?????????
o
%transform/mean_1/mean_and_var/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
#transform/mean_1/mean_and_var/Sum_1Sum$transform/mean_1/mean_and_var/Square%transform/mean_1/mean_and_var/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
'transform/mean_1/mean_and_var/truediv_1RealDiv#transform/mean_1/mean_and_var/Sum_1"transform/mean_1/mean_and_var/Cast*
T0*
_output_shapes
: 
h
#transform/mean_1/mean_and_var/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    
j
)transform/mean_1/mean_and_var/PlaceholderPlaceholder*
_output_shapes
: *
dtype0*
shape: 
l
+transform/mean_1/mean_and_var/Placeholder_1Placeholder*
_output_shapes
: *
dtype0*
shape: 
i
transform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
k
!transform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
k
!transform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
transform/strided_slice_1StridedSlice/transform/inputs/inputs/Fare/Placeholder_2_copytransform/strided_slice_1/stack!transform/strided_slice_1/stack_1!transform/strided_slice_1/stack_2*
Index0*
T0	*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
h
&transform/SparseTensor_1/dense_shape/1Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
$transform/SparseTensor_1/dense_shapePacktransform/strided_slice_1&transform/SparseTensor_1/dense_shape/1*
N*
T0	*
_output_shapes
:*

axis 
?
transform/SparseToDense_1SparseToDense-transform/inputs/inputs/Fare/Placeholder_copy$transform/SparseTensor_1/dense_shape/transform/inputs/inputs/Fare/Placeholder_1_copyConst_2*
T0*
Tindices0	*'
_output_shapes
:?????????*
validate_indices(
g
transform/IsNan_1IsNantransform/SparseToDense_1*
T0*'
_output_shapes
:?????????
?
transform/SelectV2_1SelectV2transform/IsNan_1Const_2transform/SparseToDense_1*
T0*'
_output_shapes
:?????????
y
transform/Squeeze_1Squeezetransform/SelectV2_1*
T0*#
_output_shapes
:?????????*
squeeze_dims

|
.transform/scale_to_z_score_1/mean_and_var/SizeSizetransform/Squeeze_1*
T0*
_output_shapes
: *
out_type0
?
.transform/scale_to_z_score_1/mean_and_var/CastCast.transform/scale_to_z_score_1/mean_and_var/Size*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 
y
/transform/scale_to_z_score_1/mean_and_var/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
-transform/scale_to_z_score_1/mean_and_var/SumSumtransform/Squeeze_1/transform/scale_to_z_score_1/mean_and_var/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
1transform/scale_to_z_score_1/mean_and_var/truedivRealDiv-transform/scale_to_z_score_1/mean_and_var/Sum.transform/scale_to_z_score_1/mean_and_var/Cast*
T0*
_output_shapes
: 
?
-transform/scale_to_z_score_1/mean_and_var/subSubtransform/Squeeze_11transform/scale_to_z_score_1/mean_and_var/truediv*
T0*#
_output_shapes
:?????????
?
0transform/scale_to_z_score_1/mean_and_var/SquareSquare-transform/scale_to_z_score_1/mean_and_var/sub*
T0*#
_output_shapes
:?????????
{
1transform/scale_to_z_score_1/mean_and_var/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
?
/transform/scale_to_z_score_1/mean_and_var/Sum_1Sum0transform/scale_to_z_score_1/mean_and_var/Square1transform/scale_to_z_score_1/mean_and_var/Const_1*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
3transform/scale_to_z_score_1/mean_and_var/truediv_1RealDiv/transform/scale_to_z_score_1/mean_and_var/Sum_1.transform/scale_to_z_score_1/mean_and_var/Cast*
T0*
_output_shapes
: 
t
/transform/scale_to_z_score_1/mean_and_var/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    
v
5transform/scale_to_z_score_1/mean_and_var/PlaceholderPlaceholder*
_output_shapes
: *
dtype0*
shape: 
x
7transform/scale_to_z_score_1/mean_and_var/Placeholder_1Placeholder*
_output_shapes
: *
dtype0*
shape: 
t
 transform/scale_to_z_score_1/subSubtransform/Squeeze_1Const_12*
T0*#
_output_shapes
:?????????
T
!transform/scale_to_z_score_1/SqrtSqrtConst_13*
T0*
_output_shapes
: 
l
'transform/scale_to_z_score_1/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
?
%transform/scale_to_z_score_1/NotEqualNotEqual!transform/scale_to_z_score_1/Sqrt'transform/scale_to_z_score_1/NotEqual/y*
T0*
_output_shapes
: *
incompatible_shape_error(
?
'transform/scale_to_z_score_1/zeros_like	ZerosLike transform/scale_to_z_score_1/sub*
T0*#
_output_shapes
:?????????
?
!transform/scale_to_z_score_1/CastCast%transform/scale_to_z_score_1/NotEqual*

DstT0*

SrcT0
*
Truncate( *
_output_shapes
: 
?
 transform/scale_to_z_score_1/addAddV2'transform/scale_to_z_score_1/zeros_like!transform/scale_to_z_score_1/Cast*
T0*#
_output_shapes
:?????????
?
#transform/scale_to_z_score_1/Cast_1Cast transform/scale_to_z_score_1/add*

DstT0
*

SrcT0*
Truncate( *#
_output_shapes
:?????????
?
$transform/scale_to_z_score_1/truedivRealDiv transform/scale_to_z_score_1/sub!transform/scale_to_z_score_1/Sqrt*
T0*#
_output_shapes
:?????????
?
%transform/scale_to_z_score_1/SelectV2SelectV2#transform/scale_to_z_score_1/Cast_1$transform/scale_to_z_score_1/truediv transform/scale_to_z_score_1/sub*
T0*#
_output_shapes
:?????????
i
transform/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
k
!transform/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
k
!transform/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
transform/strided_slice_2StridedSlice3transform/inputs/inputs/Embarked/Placeholder_2_copytransform/strided_slice_2/stack!transform/strided_slice_2/stack_1!transform/strided_slice_2/stack_2*
Index0*
T0	*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
h
&transform/SparseTensor_2/dense_shape/1Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
$transform/SparseTensor_2/dense_shapePacktransform/strided_slice_2&transform/SparseTensor_2/dense_shape/1*
N*
T0	*
_output_shapes
:*

axis 
h
'transform/SparseToDense_2/default_valueConst*
_output_shapes
: *
dtype0*
valueB B 
?
transform/SparseToDense_2SparseToDense1transform/inputs/inputs/Embarked/Placeholder_copy$transform/SparseTensor_2/dense_shape3transform/inputs/inputs/Embarked/Placeholder_1_copy'transform/SparseToDense_2/default_value*
T0*
Tindices0	*'
_output_shapes
:?????????*
validate_indices(
~
transform/Squeeze_2Squeezetransform/SparseToDense_2*
T0*#
_output_shapes
:?????????*
squeeze_dims

?
?transform/compute_and_apply_vocabulary/vocabulary/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
9transform/compute_and_apply_vocabulary/vocabulary/ReshapeReshapetransform/Squeeze_2?transform/compute_and_apply_vocabulary/vocabulary/Reshape/shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
stransform/compute_and_apply_vocabulary/vocabulary/vocab_compute_and_apply_vocabulary_vocabulary_unpruned_vocab_sizePlaceholder*
_output_shapes
: *
dtype0	*
shape: 
~
=transform/compute_and_apply_vocabulary/vocabulary/PlaceholderPlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
8transform/compute_and_apply_vocabulary/apply_vocab/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
=transform/compute_and_apply_vocabulary/apply_vocab/hash_tableHashTableV2*
_output_shapes
: *
	container *
	key_dtype0*y
shared_namejhhash_table_Tensor("compute_and_apply_vocabulary/vocabulary/Placeholder:0", shape=(), dtype=string)_-2_-1*
use_node_name_sharing( *
value_dtype0	
?
_transform/compute_and_apply_vocabulary/apply_vocab/text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV2=transform/compute_and_apply_vocabulary/apply_vocab/hash_tableConst_5*
	delimiter	*
	key_index?????????*
value_index?????????*

vocab_size?????????
?
dtransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Size/hash_table_Size/LookupTableSizeV2LookupTableSizeV2=transform/compute_and_apply_vocabulary/apply_vocab/hash_table*
_output_shapes
: 
?
Htransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Size/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R

?
Ftransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Size/addAddV2dtransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Size/hash_table_Size/LookupTableSizeV2Htransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Size/add/y*
T0	*
_output_shapes
: 
?
Ptransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/hash_bucketStringToHashBucketFasttransform/Squeeze_2*#
_output_shapes
:?????????*
num_buckets

?
htransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2=transform/compute_and_apply_vocabulary/apply_vocab/hash_tabletransform/Squeeze_28transform/compute_and_apply_vocabulary/apply_vocab/Const*	
Tin0*

Tout0	*#
_output_shapes
:?????????
?
ftransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/hash_table_Size/LookupTableSizeV2LookupTableSizeV2=transform/compute_and_apply_vocabulary/apply_vocab/hash_table*
_output_shapes
: 
?
Htransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/AddAddPtransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/hash_bucketftransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/hash_table_Size/LookupTableSizeV2*
T0	*#
_output_shapes
:?????????
?
Mtransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/NotEqualNotEqualhtransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/hash_table_Lookup/LookupTableFindV28transform/compute_and_apply_vocabulary/apply_vocab/Const*
T0	*#
_output_shapes
:?????????*
incompatible_shape_error(
?
Mtransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/SelectV2SelectV2Mtransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/NotEqualhtransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2Htransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/Add*
T0	*#
_output_shapes
:?????????
|
:transform/compute_and_apply_vocabulary/apply_vocab/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 
z
8transform/compute_and_apply_vocabulary/apply_vocab/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
?
6transform/compute_and_apply_vocabulary/apply_vocab/subSubFtransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Size/add8transform/compute_and_apply_vocabulary/apply_vocab/sub/y*
T0	*
_output_shapes
: 
i
transform/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 
k
!transform/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
k
!transform/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
transform/strided_slice_3StridedSlice1transform/inputs/inputs/Pclass/Placeholder_2_copytransform/strided_slice_3/stack!transform/strided_slice_3/stack_1!transform/strided_slice_3/stack_2*
Index0*
T0	*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
h
&transform/SparseTensor_3/dense_shape/1Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
$transform/SparseTensor_3/dense_shapePacktransform/strided_slice_3&transform/SparseTensor_3/dense_shape/1*
N*
T0	*
_output_shapes
:*

axis 
i
'transform/SparseToDense_3/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
transform/SparseToDense_3SparseToDense/transform/inputs/inputs/Pclass/Placeholder_copy$transform/SparseTensor_3/dense_shape1transform/inputs/inputs/Pclass/Placeholder_1_copy'transform/SparseToDense_3/default_value*
T0	*
Tindices0	*'
_output_shapes
:?????????*
validate_indices(
~
transform/Squeeze_3Squeezetransform/SparseToDense_3*
T0	*#
_output_shapes
:?????????*
squeeze_dims

?
Atransform/compute_and_apply_vocabulary_1/vocabulary/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
;transform/compute_and_apply_vocabulary_1/vocabulary/ReshapeReshapetransform/Squeeze_3Atransform/compute_and_apply_vocabulary_1/vocabulary/Reshape/shape*
T0	*
Tshape0*#
_output_shapes
:?????????
?
wtransform/compute_and_apply_vocabulary_1/vocabulary/vocab_compute_and_apply_vocabulary_1_vocabulary_unpruned_vocab_sizePlaceholder*
_output_shapes
: *
dtype0	*
shape: 
?
?transform/compute_and_apply_vocabulary_1/vocabulary/PlaceholderPlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
:transform/compute_and_apply_vocabulary_1/apply_vocab/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
?transform/compute_and_apply_vocabulary_1/apply_vocab/hash_tableHashTableV2*
_output_shapes
: *
	container *
	key_dtype0	*{
shared_nameljhash_table_Tensor("compute_and_apply_vocabulary_1/vocabulary/Placeholder:0", shape=(), dtype=string)_-2_-1*
use_node_name_sharing( *
value_dtype0	
?
atransform/compute_and_apply_vocabulary_1/apply_vocab/text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV2?transform/compute_and_apply_vocabulary_1/apply_vocab/hash_tableConst_7*
	delimiter	*
	key_index?????????*
value_index?????????*

vocab_size?????????
?
ftransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Size/hash_table_Size/LookupTableSizeV2LookupTableSizeV2?transform/compute_and_apply_vocabulary_1/apply_vocab/hash_table*
_output_shapes
: 
?
Jtransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Size/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R

?
Htransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Size/addAddV2ftransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Size/hash_table_Size/LookupTableSizeV2Jtransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Size/add/y*
T0	*
_output_shapes
: 
?
Otransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Lookup/AsStringAsStringtransform/Squeeze_3*
T0	*#
_output_shapes
:?????????*

fill *
	precision?????????*

scientific( *
shortest( *
width?????????
?
Rtransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Lookup/hash_bucketStringToHashBucketFastOtransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Lookup/AsString*#
_output_shapes
:?????????*
num_buckets

?
jtransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2?transform/compute_and_apply_vocabulary_1/apply_vocab/hash_tabletransform/Squeeze_3:transform/compute_and_apply_vocabulary_1/apply_vocab/Const*	
Tin0	*

Tout0	*#
_output_shapes
:?????????
?
htransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Lookup/hash_table_Size/LookupTableSizeV2LookupTableSizeV2?transform/compute_and_apply_vocabulary_1/apply_vocab/hash_table*
_output_shapes
: 
?
Jtransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Lookup/AddAddRtransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Lookup/hash_buckethtransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Lookup/hash_table_Size/LookupTableSizeV2*
T0	*#
_output_shapes
:?????????
?
Otransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Lookup/NotEqualNotEqualjtransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2:transform/compute_and_apply_vocabulary_1/apply_vocab/Const*
T0	*#
_output_shapes
:?????????*
incompatible_shape_error(
?
Otransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Lookup/SelectV2SelectV2Otransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Lookup/NotEqualjtransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2Jtransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Lookup/Add*
T0	*#
_output_shapes
:?????????
~
<transform/compute_and_apply_vocabulary_1/apply_vocab/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 
|
:transform/compute_and_apply_vocabulary_1/apply_vocab/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
?
8transform/compute_and_apply_vocabulary_1/apply_vocab/subSubHtransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Size/add:transform/compute_and_apply_vocabulary_1/apply_vocab/sub/y*
T0	*
_output_shapes
: 
i
transform/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 
k
!transform/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
k
!transform/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
transform/strided_slice_4StridedSlice.transform/inputs/inputs/Sex/Placeholder_2_copytransform/strided_slice_4/stack!transform/strided_slice_4/stack_1!transform/strided_slice_4/stack_2*
Index0*
T0	*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
h
&transform/SparseTensor_4/dense_shape/1Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
$transform/SparseTensor_4/dense_shapePacktransform/strided_slice_4&transform/SparseTensor_4/dense_shape/1*
N*
T0	*
_output_shapes
:*

axis 
h
'transform/SparseToDense_4/default_valueConst*
_output_shapes
: *
dtype0*
valueB B 
?
transform/SparseToDense_4SparseToDense,transform/inputs/inputs/Sex/Placeholder_copy$transform/SparseTensor_4/dense_shape.transform/inputs/inputs/Sex/Placeholder_1_copy'transform/SparseToDense_4/default_value*
T0*
Tindices0	*'
_output_shapes
:?????????*
validate_indices(
~
transform/Squeeze_4Squeezetransform/SparseToDense_4*
T0*#
_output_shapes
:?????????*
squeeze_dims

?
Atransform/compute_and_apply_vocabulary_2/vocabulary/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
;transform/compute_and_apply_vocabulary_2/vocabulary/ReshapeReshapetransform/Squeeze_4Atransform/compute_and_apply_vocabulary_2/vocabulary/Reshape/shape*
T0*
Tshape0*#
_output_shapes
:?????????
?
wtransform/compute_and_apply_vocabulary_2/vocabulary/vocab_compute_and_apply_vocabulary_2_vocabulary_unpruned_vocab_sizePlaceholder*
_output_shapes
: *
dtype0	*
shape: 
?
?transform/compute_and_apply_vocabulary_2/vocabulary/PlaceholderPlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
:transform/compute_and_apply_vocabulary_2/apply_vocab/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????
?
?transform/compute_and_apply_vocabulary_2/apply_vocab/hash_tableHashTableV2*
_output_shapes
: *
	container *
	key_dtype0*{
shared_nameljhash_table_Tensor("compute_and_apply_vocabulary_2/vocabulary/Placeholder:0", shape=(), dtype=string)_-2_-1*
use_node_name_sharing( *
value_dtype0	
?
atransform/compute_and_apply_vocabulary_2/apply_vocab/text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV2?transform/compute_and_apply_vocabulary_2/apply_vocab/hash_tableConst_9*
	delimiter	*
	key_index?????????*
value_index?????????*

vocab_size?????????
?
ftransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Size/hash_table_Size/LookupTableSizeV2LookupTableSizeV2?transform/compute_and_apply_vocabulary_2/apply_vocab/hash_table*
_output_shapes
: 
?
Jtransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Size/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R

?
Htransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Size/addAddV2ftransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Size/hash_table_Size/LookupTableSizeV2Jtransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Size/add/y*
T0	*
_output_shapes
: 
?
Rtransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Lookup/hash_bucketStringToHashBucketFasttransform/Squeeze_4*#
_output_shapes
:?????????*
num_buckets

?
jtransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2?transform/compute_and_apply_vocabulary_2/apply_vocab/hash_tabletransform/Squeeze_4:transform/compute_and_apply_vocabulary_2/apply_vocab/Const*	
Tin0*

Tout0	*#
_output_shapes
:?????????
?
htransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Lookup/hash_table_Size/LookupTableSizeV2LookupTableSizeV2?transform/compute_and_apply_vocabulary_2/apply_vocab/hash_table*
_output_shapes
: 
?
Jtransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Lookup/AddAddRtransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Lookup/hash_buckethtransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Lookup/hash_table_Size/LookupTableSizeV2*
T0	*#
_output_shapes
:?????????
?
Otransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Lookup/NotEqualNotEqualjtransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2:transform/compute_and_apply_vocabulary_2/apply_vocab/Const*
T0	*#
_output_shapes
:?????????*
incompatible_shape_error(
?
Otransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Lookup/SelectV2SelectV2Otransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Lookup/NotEqualjtransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2Jtransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Lookup/Add*
T0	*#
_output_shapes
:?????????
~
<transform/compute_and_apply_vocabulary_2/apply_vocab/Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 
|
:transform/compute_and_apply_vocabulary_2/apply_vocab/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R
?
8transform/compute_and_apply_vocabulary_2/apply_vocab/subSubHtransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Size/add:transform/compute_and_apply_vocabulary_2/apply_vocab/sub/y*
T0	*
_output_shapes
: 
l
transform/ConstConst*
_output_shapes

:*
dtype0*%
valueB"      ??   @
i
transform/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB: 
k
!transform/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
k
!transform/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
transform/strided_slice_5StridedSlice0transform/inputs/inputs/Parch/Placeholder_2_copytransform/strided_slice_5/stack!transform/strided_slice_5/stack_1!transform/strided_slice_5/stack_2*
Index0*
T0	*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
h
&transform/SparseTensor_5/dense_shape/1Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
$transform/SparseTensor_5/dense_shapePacktransform/strided_slice_5&transform/SparseTensor_5/dense_shape/1*
N*
T0	*
_output_shapes
:*

axis 
i
'transform/SparseToDense_5/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
transform/SparseToDense_5SparseToDense.transform/inputs/inputs/Parch/Placeholder_copy$transform/SparseTensor_5/dense_shape0transform/inputs/inputs/Parch/Placeholder_1_copy'transform/SparseToDense_5/default_value*
T0	*
Tindices0	*'
_output_shapes
:?????????*
validate_indices(
~
transform/Squeeze_5Squeezetransform/SparseToDense_5*
T0	*#
_output_shapes
:?????????*
squeeze_dims

\
transform/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :
l
transform/assert_rank/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
L
Dtransform/assert_rank/assert_type/statically_determined_correct_typeNoOp
=
5transform/assert_rank/static_checks_determined_all_okNoOp
?
6transform/apply_buckets/assign_buckets_all_shapes/CastCasttransform/Squeeze_5*

DstT0*

SrcT0	*
Truncate( *#
_output_shapes
:?????????
?
Ftransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
?
Ttransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
Vtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Vtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Ntransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/strided_sliceStridedSliceFtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/ShapeTtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/strided_slice/stackVtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/strided_slice/stack_1Vtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
?
Etransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/CastCastNtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/strided_slice*

DstT0	*

SrcT0*
Truncate( *
_output_shapes
: 
?
Dtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/NegNegtransform/Const*
T0*
_output_shapes

:
?
Otransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Jtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/ReverseV2	ReverseV2Dtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/NegOtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/ReverseV2/axis*
T0*

Tidx0*
_output_shapes

:
?
Ftransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/Neg_1Neg6transform/apply_buckets/assign_buckets_all_shapes/Cast*
T0*#
_output_shapes
:?????????
?
Ftransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
Dtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/MaxMaxFtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/Neg_1Ftransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
Rtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/concat/values_1/0PackDtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/Max*
N*
T0*
_output_shapes
:*

axis 
?
Ptransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/concat/values_1PackRtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/concat/values_1/0*
N*
T0*
_output_shapes

:*

axis 
?
Ltransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
Gtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/concatConcatV2Jtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/ReverseV2Ptransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/concat/values_1Ltransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/concat/axis*
N*
T0*

Tidx0*
_output_shapes

:
?
Ftransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/Neg_2Neg6transform/apply_buckets/assign_buckets_all_shapes/Cast*
T0*#
_output_shapes
:?????????
?
Htransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/unstackUnpackGtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/concat*
T0*
_output_shapes
:*

axis *	
num
?
Vtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/BoostedTreesBucketizeBoostedTreesBucketizeFtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/Neg_2Htransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/unstack*#
_output_shapes
:?????????*
num_features
?
Gtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/Cast_1CastVtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/BoostedTreesBucketize*

DstT0	*

SrcT0*
Truncate( *#
_output_shapes
:?????????
?
Dtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/SubSubEtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/CastGtransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/Cast_1*
T0	*#
_output_shapes
:?????????
_
transform/apply_buckets/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
n
transform/apply_buckets/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
u
+transform/apply_buckets/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
w
-transform/apply_buckets/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
w
-transform/apply_buckets/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
%transform/apply_buckets/strided_sliceStridedSlicetransform/apply_buckets/Shape+transform/apply_buckets/strided_slice/stack-transform/apply_buckets/strided_slice/stack_1-transform/apply_buckets/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
n
transform/Const_1Const*
_output_shapes

:*
dtype0*%
valueB"      ??   @
i
transform/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 
k
!transform/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
k
!transform/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
transform/strided_slice_6StridedSlice0transform/inputs/inputs/SibSp/Placeholder_2_copytransform/strided_slice_6/stack!transform/strided_slice_6/stack_1!transform/strided_slice_6/stack_2*
Index0*
T0	*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
h
&transform/SparseTensor_6/dense_shape/1Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
$transform/SparseTensor_6/dense_shapePacktransform/strided_slice_6&transform/SparseTensor_6/dense_shape/1*
N*
T0	*
_output_shapes
:*

axis 
i
'transform/SparseToDense_6/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
transform/SparseToDense_6SparseToDense.transform/inputs/inputs/SibSp/Placeholder_copy$transform/SparseTensor_6/dense_shape0transform/inputs/inputs/SibSp/Placeholder_1_copy'transform/SparseToDense_6/default_value*
T0	*
Tindices0	*'
_output_shapes
:?????????*
validate_indices(
~
transform/Squeeze_6Squeezetransform/SparseToDense_6*
T0	*#
_output_shapes
:?????????*
squeeze_dims

^
transform/assert_rank_1/rankConst*
_output_shapes
: *
dtype0*
value	B :
n
transform/assert_rank_1/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
N
Ftransform/assert_rank_1/assert_type/statically_determined_correct_typeNoOp
?
7transform/assert_rank_1/static_checks_determined_all_okNoOp
?
8transform/apply_buckets_1/assign_buckets_all_shapes/CastCasttransform/Squeeze_6*

DstT0*

SrcT0	*
Truncate( *#
_output_shapes
:?????????
?
Htransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
?
Vtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
?
Xtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Xtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
Ptransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/strided_sliceStridedSliceHtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/ShapeVtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/strided_slice/stackXtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/strided_slice/stack_1Xtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
?
Gtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/CastCastPtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/strided_slice*

DstT0	*

SrcT0*
Truncate( *
_output_shapes
: 
?
Ftransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/NegNegtransform/Const_1*
T0*
_output_shapes

:
?
Qtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
Ltransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/ReverseV2	ReverseV2Ftransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/NegQtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/ReverseV2/axis*
T0*

Tidx0*
_output_shapes

:
?
Htransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/Neg_1Neg8transform/apply_buckets_1/assign_buckets_all_shapes/Cast*
T0*#
_output_shapes
:?????????
?
Htransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
Ftransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/MaxMaxHtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/Neg_1Htransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
?
Ttransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/concat/values_1/0PackFtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/Max*
N*
T0*
_output_shapes
:*

axis 
?
Rtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/concat/values_1PackTtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/concat/values_1/0*
N*
T0*
_output_shapes

:*

axis 
?
Ntransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
Itransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/concatConcatV2Ltransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/ReverseV2Rtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/concat/values_1Ntransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/concat/axis*
N*
T0*

Tidx0*
_output_shapes

:
?
Htransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/Neg_2Neg8transform/apply_buckets_1/assign_buckets_all_shapes/Cast*
T0*#
_output_shapes
:?????????
?
Jtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/unstackUnpackItransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/concat*
T0*
_output_shapes
:*

axis *	
num
?
Xtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/BoostedTreesBucketizeBoostedTreesBucketizeHtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/Neg_2Jtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/unstack*#
_output_shapes
:?????????*
num_features
?
Itransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/Cast_1CastXtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/BoostedTreesBucketize*

DstT0	*

SrcT0*
Truncate( *#
_output_shapes
:?????????
?
Ftransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/SubSubGtransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/CastItransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/Cast_1*
T0	*#
_output_shapes
:?????????
a
transform/apply_buckets_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
p
transform/apply_buckets_1/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
w
-transform/apply_buckets_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
y
/transform/apply_buckets_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
y
/transform/apply_buckets_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
'transform/apply_buckets_1/strided_sliceStridedSlicetransform/apply_buckets_1/Shape-transform/apply_buckets_1/strided_slice/stack/transform/apply_buckets_1/strided_slice/stack_1/transform/apply_buckets_1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
i
transform/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB: 
k
!transform/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
k
!transform/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
transform/strided_slice_7StridedSlice3transform/inputs/inputs/Survived/Placeholder_2_copytransform/strided_slice_7/stack!transform/strided_slice_7/stack_1!transform/strided_slice_7/stack_2*
Index0*
T0	*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
h
&transform/SparseTensor_7/dense_shape/1Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
$transform/SparseTensor_7/dense_shapePacktransform/strided_slice_7&transform/SparseTensor_7/dense_shape/1*
N*
T0	*
_output_shapes
:*

axis 
i
'transform/SparseToDense_7/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R 
?
transform/SparseToDense_7SparseToDense1transform/inputs/inputs/Survived/Placeholder_copy$transform/SparseTensor_7/dense_shape3transform/inputs/inputs/Survived/Placeholder_1_copy'transform/SparseToDense_7/default_value*
T0	*
Tindices0	*'
_output_shapes
:?????????*
validate_indices(
~
transform/Squeeze_7Squeezetransform/SparseToDense_7*
T0	*#
_output_shapes
:?????????*
squeeze_dims


transform/initNoOp

transform/init_1NoOp

transform/init_2NoOp

initNoOp"?"6
asset_filepaths#
!
	Const_5:0
	Const_7:0
	Const_9:0"?
saved_model_assets?*?
k
+type.googleapis.com/tensorflow.AssetFileDef<

	Const_5:0-vocab_compute_and_apply_vocabulary_vocabulary
m
+type.googleapis.com/tensorflow.AssetFileDef>

	Const_7:0/vocab_compute_and_apply_vocabulary_1_vocabulary
m
+type.googleapis.com/tensorflow.AssetFileDef>

	Const_9:0/vocab_compute_and_apply_vocabulary_2_vocabulary"?
table_initializer?
?
_transform/compute_and_apply_vocabulary/apply_vocab/text_file_init/InitializeTableFromTextFileV2
atransform/compute_and_apply_vocabulary_1/apply_vocab/text_file_init/InitializeTableFromTextFileV2
atransform/compute_and_apply_vocabulary_2/apply_vocab/text_file_init/InitializeTableFromTextFileV2"?
tft_schema_override_max?
?
8transform/compute_and_apply_vocabulary/apply_vocab/sub:0
:transform/compute_and_apply_vocabulary_1/apply_vocab/sub:0
:transform/compute_and_apply_vocabulary_2/apply_vocab/sub:0
'transform/apply_buckets/strided_slice:0
)transform/apply_buckets_1/strided_slice:0"?
tft_schema_override_min?
?
<transform/compute_and_apply_vocabulary/apply_vocab/Const_1:0
>transform/compute_and_apply_vocabulary_1/apply_vocab/Const_1:0
>transform/compute_and_apply_vocabulary_2/apply_vocab/Const_1:0
transform/apply_buckets/Const:0
!transform/apply_buckets_1/Const:0"?
tft_schema_override_tensor?
?
Otransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/SelectV2:0
Qtransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Lookup/SelectV2:0
Qtransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Lookup/SelectV2:0
Ftransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/Sub:0
Htransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/Sub:0*?
transform_signature?
?
Age???????????????????"p
$transform/inputs/Age/Placeholder_1:0"transform/inputs/Age/Placeholder:0$transform/inputs/Age/Placeholder_2:0
?
Cabin???????????????????"v
&transform/inputs/Cabin/Placeholder_1:0$transform/inputs/Cabin/Placeholder:0&transform/inputs/Cabin/Placeholder_2:0
?
Embarked???????????????????"
)transform/inputs/Embarked/Placeholder_1:0'transform/inputs/Embarked/Placeholder:0)transform/inputs/Embarked/Placeholder_2:0
?
Fare???????????????????"s
%transform/inputs/Fare/Placeholder_1:0#transform/inputs/Fare/Placeholder:0%transform/inputs/Fare/Placeholder_2:0
?
Name???????????????????"s
%transform/inputs/Name/Placeholder_1:0#transform/inputs/Name/Placeholder:0%transform/inputs/Name/Placeholder_2:0
?
Parch?	??????????????????"v
&transform/inputs/Parch/Placeholder_1:0$transform/inputs/Parch/Placeholder:0&transform/inputs/Parch/Placeholder_2:0
?
PassengerId?	??????????????????"?
,transform/inputs/PassengerId/Placeholder_1:0*transform/inputs/PassengerId/Placeholder:0,transform/inputs/PassengerId/Placeholder_2:0
?
Pclass?	??????????????????"y
'transform/inputs/Pclass/Placeholder_1:0%transform/inputs/Pclass/Placeholder:0'transform/inputs/Pclass/Placeholder_2:0
?
Sex???????????????????"p
$transform/inputs/Sex/Placeholder_1:0"transform/inputs/Sex/Placeholder:0$transform/inputs/Sex/Placeholder_2:0
?
SibSp?	??????????????????"v
&transform/inputs/SibSp/Placeholder_1:0$transform/inputs/SibSp/Placeholder:0&transform/inputs/SibSp/Placeholder_2:0
?
Survived?	??????????????????"
)transform/inputs/Survived/Placeholder_1:0'transform/inputs/Survived/Placeholder:0)transform/inputs/Survived/Placeholder_2:0
?
Ticket???????????????????"y
'transform/inputs/Ticket/Placeholder_1:0%transform/inputs/Ticket/Placeholder:0'transform/inputs/Ticket/Placeholder_2:0B
Age_xf8
%transform/scale_to_z_score/SelectV2:0?????????q
Embarked_xfb
Otransform/compute_and_apply_vocabulary/apply_vocab/hash_table_Lookup/SelectV2:0	?????????E
Fare_xf:
'transform/scale_to_z_score_1/SelectV2:0?????????e
Parch_xfY
Ftransform/apply_buckets/assign_buckets_all_shapes/assign_buckets/Sub:0	?????????q
	Pclass_xfd
Qtransform/compute_and_apply_vocabulary_1/apply_vocab/hash_table_Lookup/SelectV2:0	?????????n
Sex_xfd
Qtransform/compute_and_apply_vocabulary_2/apply_vocab/hash_table_Lookup/SelectV2:0	?????????g
SibSp_xf[
Htransform/apply_buckets_1/assign_buckets_all_shapes/assign_buckets/Sub:0	?????????7
Survived_xf(
transform/Squeeze_7:0	?????????tensorflow/serving/predict