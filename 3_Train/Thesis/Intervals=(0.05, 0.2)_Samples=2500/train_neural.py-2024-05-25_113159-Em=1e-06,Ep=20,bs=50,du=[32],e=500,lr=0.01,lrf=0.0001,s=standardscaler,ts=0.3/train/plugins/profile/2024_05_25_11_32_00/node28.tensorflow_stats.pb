"�9
BHostIDLE"IDLE1���Sck�@A���Sck�@a)�	̏��?i)�	̏��?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1��Q��\@9��Q��\@A��Q��\@I��Q��\@a�]���*�?i�ܥ���?�Unknown�
tHostMatMul" gradient_tape/model/dense/MatMul(1X9���D@9X9���D@AX9���D@IX9���D@ak��<��?iU$���^�?�Unknown
jHost_FusedMatMul"model/dense/Relu(1�MbX�D@9�MbX�D@A�MbX�D@I�MbX�D@a7�k�}�?i�j���?�Unknown
iHostWriteSummary"WriteSummary(1+���B@9+���B@A+���B@I+���B@a[1��?ir��#��?�Unknown�
rHostDataset"Iterator::Root::ParallelMapV2(1NbX9�@@9NbX9�@@ANbX9�@@INbX9�@@a��sP�ܒ?i��b�]�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1/�$��<@9/�$��<@A/�$��<@I/�$��<@a�P{�63�?il�����?�Unknown
�HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1�$��=@9�$��=@A��S㥛9@I��S㥛9@a;	C��?i��-�S�?�Unknown
�	HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1��v���@@9��v���@@AbX9��7@IbX9��7@a�v	]��?i�6Oa��?�Unknown
x
HostMatMul"$gradient_tape/model/dense_1/MatMul_1(1/�$�7@9/�$�7@A/�$�7@I/�$�7@a��!�s*�?i�>�'�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1㥛� 7@9㥛� 7@A㥛� 7@I㥛� 7@a/���[$�?i�������?�Unknown
rHostWriteSummary"batch_loss/write_summary(1+����6@9+����6@A+����6@I+����6@aO�(��?i�a�.��?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1�rh��<4@9�rh��<4@A�rh��<4@I�rh��<4@ayG�����?i4�A�R�?�Unknown
vHostMatMul""gradient_tape/model/dense_1/MatMul(1h��|?�3@9h��|?�3@Ah��|?�3@Ih��|?�3@a��<�[��?i�~�n��?�Unknown
oHost_FusedMatMul"model/dense_1/BiasAdd(15^�Ib1@95^�Ib1@A5^�Ib1@I5^�Ib1@a��H�&��?i�7?��?�Unknown
lHostIteratorGetNext"IteratorGetNext(1^�Ik.@9^�Ik.@A^�Ik.@I^�Ik.@a��N~N=�?i��0W4A�?�Unknown
cHostDataset"Iterator::Root(1�p=
��G@9�p=
��G@AX9��v~,@IX9��v~,@aZ���'&�?i�	��́�?�Unknown
^HostGatherV2"GatherV2(1����x�+@9����x�+@A����x�+@I����x�+@aH��m�?in�����?�Unknown
[HostPow"
Adam/Pow_1(1)\���h*@9)\���h*@A)\���h*@I)\���h*@a�M�$��}?i
������?�Unknown
[HostCast"	Adam/Cast(1-���g(@9-���g(@A-���g(@I-���g(@a4�+9�{?i��QG4�?�Unknown
�HostBiasAddGrad"/gradient_tape/model/dense_1/BiasAdd/BiasAddGrad(1��~j�4(@9��~j�4(@A��~j�4(@I��~j�4(@a��6%p{?i�Z�/'k�?�Unknown
�HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1��ʡEv&@9��ʡEv&@A��ʡEv&@I��ʡEv&@a=��l�uy?iG>v��?�Unknown
�HostBiasAddGrad"-gradient_tape/model/dense/BiasAdd/BiasAddGrad(1y�&1l&@9y�&1l&@Ay�&1l&@Iy�&1l&@a ��[jy?i�Y{����?�Unknown
[HostAddV2"Adam/add(1�(\���%@9�(\���%@A�(\���%@I�(\���%@a�{�(K�x?i�6�h��?�Unknown
xHostReluGrad""gradient_tape/model/dense/ReluGrad(1{�G�z%@9{�G�z%@A{�G�z%@I{�G�z%@a{�;,Yx?i.�%mb3�?�Unknown
~HostReadVariableOp""model/dense/BiasAdd/ReadVariableOp(1
ףp=
%@9
ףp=
%@A
ףp=
%@I
ףp=
%@a8Z�T�w?i�Yc�?�Unknown
eHost
LogicalAnd"
LogicalAnd(1R����$@9R����$@AR����$@IR����$@a ��}�w?i"�H��?�Unknown�
�HostSquaredDifference"$mean_squared_error/SquaredDifference(1ffffff$@9ffffff$@Affffff$@Iffffff$@acȩ�w?i�h�M���?�Unknown
�HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1+��#@9+��#@A+��#@I+��#@a����6v?i����?�Unknown
}HostReadVariableOp"!model/dense/MatMul/ReadVariableOp(1��Mb�!@9��Mb�!@A��Mb�!@I��Mb�!@aU�%a�s?iȢQ���?�Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1�&1�!@9�&1�!@A�&1�!@I�&1�!@a-��es?i���;�?�Unknown
m HostMinimum"Adam/CosineDecay/Minimum(1�Zd;� @9�Zd;� @A�Zd;� @I�Zd;� @a}���s?iGD���a�?�Unknown
v!HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1Zd;��@9Zd;��@AZd;��@IZd;��@aN��Bi�o?i �i��?�Unknown
w"HostDataset""Iterator::Root::ParallelMapV2::Zip(1��Q�>Q@9��Q�>Q@A�x�&1�@I�x�&1�@a��G��n?i�Z�|��?�Unknown
Y#HostPow"Adam/Pow(1F�����@9F�����@AF�����@IF�����@a
v�$|pm?i'���?�Unknown
u$HostSub"$gradient_tape/mean_squared_error/sub(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@a^���fMh?i����9��?�Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1��Q�@9��Q�@A��Q�@I��Q�@a�8�
^�f?i���E��?�Unknown
u&HostSum"$mean_squared_error/weighted_loss/Sum(1L7�A`e@9L7�A`e@AL7�A`e@IL7�A`e@a�g��d?iB��_� �?�Unknown
i'HostMean"mean_squared_error/Mean(1B`��"[@9B`��"[@AB`��"[@IB`��"[@a�.u�~�d?iq9�ް�?�Unknown
`(HostGatherV2"
GatherV2_1(1�E����@9�E����@A�E����@I�E����@a`{W�Cc?i�/"�(�?�Unknown
])HostCast"Adam/Cast_1(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@a��"X$�`?iγ�F�9�?�Unknown
�*HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1sh��|?@9sh��|?�?Ash��|?@Ish��|?�?a�`M��^?i�>���H�?�Unknown
e+HostCos"Adam/CosineDecay/Cos(1��S㥛@9��S㥛@A��S㥛@I��S㥛@a;�?߮�[?i����V�?�Unknown
o,HostReadVariableOp"Adam/ReadVariableOp(1+����@9+����@A+����@I+����@af�8�*[?i�\:z�d�?�Unknown
~-HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1�Q���@9�Q���@A�Q���@I�Q���@a��S"��Y?iȆ��~q�?�Unknown
`.HostDivNoNan"
div_no_nan(1+��@9+��@A+��@I+��@a1����Y?i���q~�?�Unknown
/HostReadVariableOp"#model/dense_1/MatMul/ReadVariableOp(1��Q�@9��Q�@A��Q�@I��Q�@aP`ҸY?ic�A���?�Unknown
u0HostReadVariableOp"div_no_nan/ReadVariableOp(1)\���(@9)\���(@A)\���(@I)\���(@a��&�W?ifV�T���?�Unknown
e1HostMul"Adam/CosineDecay/mul(1�z�G�@9�z�G�@A�z�G�@I�z�G�@a�ᒌ��V?iן��=��?�Unknown
w2HostMul"&gradient_tape/mean_squared_error/mul_1(1�E����@9�E����@A�E����@I�E����@a��u�XU?i)�� ��?�Unknown
|3HostDivNoNan"&mean_squared_error/weighted_loss/value(1�ʡE��@9�ʡE��@A�ʡE��@I�ʡE��@a��MQCYT?i<����?�Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_1(1�Zd;@9�Zd;@A�Zd;@I�Zd;@a��A�U�S?i�\�����?�Unknown
g5HostAddV2"Adam/CosineDecay/add(1���x�&@9���x�&@A���x�&@I���x�&@aKDyqS?i�]���?�Unknown
a6HostIdentity"Identity(1�O��n@9�O��n@A�O��n@I�O��n@aҰ,�YS?i�qQ@��?�Unknown�
T7HostMul"Mul(1?5^�I�?9?5^�I�?A?5^�I�?I?5^�I�?a����O?i��?
3��?�Unknown
v8HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1�|?5^��?9�|?5^��?A�|?5^��?I�|?5^��?a����S�J?iRU._���?�Unknown
w9HostReadVariableOp"div_no_nan/ReadVariableOp_1(1;�O��n�?9;�O��n�?A;�O��n�?I;�O��n�?ag�&�AmI?i ���G��?�Unknown
t:HostReadVariableOp"Adam/Cast/ReadVariableOp(1+�����?9+�����?A+�����?I+�����?a��hq��H?i;�W���?�Unknown
�;HostReadVariableOp"$model/dense_1/BiasAdd/ReadVariableOp(11�Zd�?91�Zd�?A1�Zd�?I1�Zd�?a��eG�E?i�ғ ��?�Unknown
�<HostReadVariableOp"'batch_loss/write_summary/ReadVariableOp(1��S㥛�?9��S㥛�?A��S㥛�?I��S㥛�?a0���E?i8�5F��?�Unknown
g=HostMul"Adam/CosineDecay/Mul_3(1/�$��?9/�$��?A/�$��?I/�$��?al����0?i�3x]��?�Unknown
i>HostAddV2"Adam/CosineDecay/add_1(1/�$���?9/�$���?A/�$���?I/�$���?aY-1@D/?i�F{`Q��?�Unknown
g?HostMul"Adam/CosineDecay/mul_2(1X9��v��?9X9��v��?AX9��v��?IX9��v��?a\��K��*?i�������?�Unknown*�8
uHostFlushSummaryWriter"FlushSummaryWriter(1��Q��\@9��Q��\@A��Q��\@I��Q��\@a����)R�?i����)R�?�Unknown�
tHostMatMul" gradient_tape/model/dense/MatMul(1X9���D@9X9���D@AX9���D@IX9���D@a}�� �:�?i�I���`�?�Unknown
jHost_FusedMatMul"model/dense/Relu(1�MbX�D@9�MbX�D@A�MbX�D@I�MbX�D@a�{��?i�K����?�Unknown
iHostWriteSummary"WriteSummary(1+���B@9+���B@A+���B@I+���B@au_FA��?i��#��?�Unknown�
rHostDataset"Iterator::Root::ParallelMapV2(1NbX9�@@9NbX9�@@ANbX9�@@INbX9�@@a(.aw?i�������?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1/�$��<@9/�$��<@A/�$��<@I/�$��<@ac+&-\�?i,f�>'�?�Unknown
�HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1�$��=@9�$��=@A��S㥛9@I��S㥛9@ad�+X�?i����R�?�Unknown
�HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1��v���@@9��v���@@AbX9��7@IbX9��7@ay�k�?i�bvP�?�Unknown
x	HostMatMul"$gradient_tape/model/dense_1/MatMul_1(1/�$�7@9/�$�7@A/�$�7@I/�$�7@a�'�dE�?i,�|oD�?�Unknown
�
HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1㥛� 7@9㥛� 7@A㥛� 7@I㥛� 7@a\���=�?i,�%��?�Unknown
rHostWriteSummary"batch_loss/write_summary(1+����6@9+����6@A+����6@I+����6@aˏ�w}�?i���걓�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1�rh��<4@9�rh��<4@A�rh��<4@I�rh��<4@a��� j�?i�"��o�?�Unknown
vHostMatMul""gradient_tape/model/dense_1/MatMul(1h��|?�3@9h��|?�3@Ah��|?�3@Ih��|?�3@a�G��	�?i�t91KG�?�Unknown
oHost_FusedMatMul"model/dense_1/BiasAdd(15^�Ib1@95^�Ib1@A5^�Ib1@I5^�Ib1@a�����?i6��ū�?�Unknown
lHostIteratorGetNext"IteratorGetNext(1^�Ik.@9^�Ik.@A^�Ik.@I^�Ik.@a��Qd,��?i�)}��?�Unknown
cHostDataset"Iterator::Root(1�p=
��G@9�p=
��G@AX9��v~,@IX9��v~,@ape	?�L�?i�f��B�?�Unknown
^HostGatherV2"GatherV2(1����x�+@9����x�+@A����x�+@I����x�+@a�삊��?iM~\���?�Unknown
[HostPow"
Adam/Pow_1(1)\���h*@9)\���h*@A)\���h*@I)\���h*@a��:�?iy��8i�?�Unknown
[HostCast"	Adam/Cast(1-���g(@9-���g(@A-���g(@I-���g(@a�?s����?ix���v��?�Unknown
�HostBiasAddGrad"/gradient_tape/model/dense_1/BiasAdd/BiasAddGrad(1��~j�4(@9��~j�4(@A��~j�4(@I��~j�4(@a#~�Qe�?iiz]q�p�?�Unknown
�HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1��ʡEv&@9��ʡEv&@A��ʡEv&@I��ʡEv&@a�*�]m�?i�"2�T��?�Unknown
�HostBiasAddGrad"-gradient_tape/model/dense/BiasAdd/BiasAddGrad(1y�&1l&@9y�&1l&@Ay�&1l&@Iy�&1l&@a���~_�?i�n���c�?�Unknown
[HostAddV2"Adam/add(1�(\���%@9�(\���%@A�(\���%@I�(\���%@a�^˘H��?iY�P���?�Unknown
xHostReluGrad""gradient_tape/model/dense/ReluGrad(1{�G�z%@9{�G�z%@A{�G�z%@I{�G�z%@aP#�L��?i�t�X3O�?�Unknown
~HostReadVariableOp""model/dense/BiasAdd/ReadVariableOp(1
ףp=
%@9
ףp=
%@A
ףp=
%@I
ףp=
%@a,�?��?i�T�T4��?�Unknown
eHost
LogicalAnd"
LogicalAnd(1R����$@9R����$@AR����$@IR����$@a��d�3�?i��/&2�?�Unknown�
�HostSquaredDifference"$mean_squared_error/SquaredDifference(1ffffff$@9ffffff$@Affffff$@Iffffff$@a�]��N��?i+ҵ`���?�Unknown
�HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1+��#@9+��#@A+��#@I+��#@aa[��0��?i�;#�
�?�Unknown
}HostReadVariableOp"!model/dense/MatMul/ReadVariableOp(1��Mb�!@9��Mb�!@A��Mb�!@I��Mb�!@aP[>>Շ?i٨j�?�Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1�&1�!@9�&1�!@A�&1�!@I�&1�!@a" (e.�?iZI�}���?�Unknown
mHostMinimum"Adam/CosineDecay/Minimum(1�Zd;� @9�Zd;� @A�Zd;� @I�Zd;� @am�?�چ?id���6"�?�Unknown
v HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1Zd;��@9Zd;��@AZd;��@IZd;��@a���Qւ?i��=�m�?�Unknown
w!HostDataset""Iterator::Root::ParallelMapV2::Zip(1��Q�>Q@9��Q�>Q@A�x�&1�@I�x�&1�@a0g$]a��?i0��q��?�Unknown
Y"HostPow"Adam/Pow(1F�����@9F�����@AF�����@IF�����@a0��0H��?i�������?�Unknown
u#HostSub"$gradient_tape/mean_squared_error/sub(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@a	_F�
}?i�qe��5�?�Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_2(1��Q�@9��Q�@A��Q�@I��Q�@a��$u5A{?iJ�O9gl�?�Unknown
u%HostSum"$mean_squared_error/weighted_loss/Sum(1L7�A`e@9L7�A`e@AL7�A`e@IL7�A`e@aNRGJ\�x?i�I��=��?�Unknown
i&HostMean"mean_squared_error/Mean(1B`��"[@9B`��"[@AB`��"[@IB`��"[@a8�C}�x?i_*l����?�Unknown
`'HostGatherV2"
GatherV2_1(1�E����@9�E����@A�E����@I�E����@a$h�8��v?i/�����?�Unknown
](HostCast"Adam/Cast_1(1�~j�t�@9�~j�t�@A�~j�t�@I�~j�t�@a�B0�-t?i��F�%�?�Unknown
�)HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1sh��|?@9sh��|?�?Ash��|?@Ish��|?�?a>_	�tr?irD�T�J�?�Unknown
e*HostCos"Adam/CosineDecay/Cos(1��S㥛@9��S㥛@A��S㥛@I��S㥛@a���7Ǫp?ij���k�?�Unknown
o+HostReadVariableOp"Adam/ReadVariableOp(1+����@9+����@A+����@I+����@a�)#�;p?i�B �Y��?�Unknown
~,HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1�Q���@9�Q���@A�Q���@I�Q���@a���01o?i�8Q�e��?�Unknown
`-HostDivNoNan"
div_no_nan(1+��@9+��@A+��@I+��@as�X9�n?i���X��?�Unknown
.HostReadVariableOp"#model/dense_1/MatMul/ReadVariableOp(1��Q�@9��Q�@A��Q�@I��Q�@a��'���m?i�\�O��?�Unknown
u/HostReadVariableOp"div_no_nan/ReadVariableOp(1)\���(@9)\���(@A)\���(@I)\���(@aQ�ܩl?iv���?�Unknown
e0HostMul"Adam/CosineDecay/mul(1�z�G�@9�z�G�@A�z�G�@I�z�G�@a�O��j?i'7Ň��?�Unknown
w1HostMul"&gradient_tape/mean_squared_error/mul_1(1�E����@9�E����@A�E����@I�E����@aj��u��i?i� ;j9�?�Unknown
|2HostDivNoNan"&mean_squared_error/weighted_loss/value(1�ʡE��@9�ʡE��@A�ʡE��@I�ʡE��@a#HM�cQh?i�M%z�Q�?�Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_1(1�Zd;@9�Zd;@A�Zd;@I�Zd;@a�n-x�Wg?ih{�/i�?�Unknown
g4HostAddV2"Adam/CosineDecay/add(1���x�&@9���x�&@A���x�&@I���x�&@am:k�;g?i��'O��?�Unknown
a5HostIdentity"Identity(1�O��n@9�O��n@A�O��n@I�O��n@a��^9 g?i��g`o��?�Unknown�
T6HostMul"Mul(1?5^�I�?9?5^�I�?A?5^�I�?I?5^�I�?af}H=�b?iI��n��?�Unknown
v7HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1�|?5^��?9�|?5^��?A�|?5^��?I�|?5^��?a���1`?i�f�π��?�Unknown
w8HostReadVariableOp"div_no_nan/ReadVariableOp_1(1;�O��n�?9;�O��n�?A;�O��n�?I;�O��n�?a�h��b^?iM��J���?�Unknown
t9HostReadVariableOp"Adam/Cast/ReadVariableOp(1+�����?9+�����?A+�����?I+�����?a�0C��]?i�<aR���?�Unknown
�:HostReadVariableOp"$model/dense_1/BiasAdd/ReadVariableOp(11�Zd�?91�Zd�?A1�Zd�?I1�Zd�?ai����DZ?ib�Ȳ���?�Unknown
�;HostReadVariableOp"'batch_loss/write_summary/ReadVariableOp(1��S㥛�?9��S㥛�?A��S㥛�?I��S㥛�?aѐ���4Y?i��$#P��?�Unknown
g<HostMul"Adam/CosineDecay/Mul_3(1/�$��?9/�$��?A/�$��?I/�$��?aK��C?iO��?�Unknown
i=HostAddV2"Adam/CosineDecay/add_1(1/�$���?9/�$���?A/�$���?I/�$���?aq��VɮB?i�Z�����?�Unknown
g>HostMul"Adam/CosineDecay/mul_2(1X9��v��?9X9��v��?AX9��v��?IX9��v��?a���&�@?i      �?�Unknown2CPU