output="runs"
device="cpu"

if [ "$1" == "hs" ]; then
	# hs dataset
	echo "run trained model for hs"
	dataset="data/hs.freq3.pre_suf.unary_closure.bin"
	model="model.hs_unary_closure_top20_word128_encoder256_rule128_node64.beam15.adadelta.simple_trans.8e39832.iter5600.npz"
	commandline="-decode_max_time_step 350 -rule_embed_dim 128 -node_embed_dim 64"
	datatype="hs"
else
	# django dataset
	echo "run trained model for django"
	dataset="data/django.cleaned.dataset.freq5.par_info.refact.space_only.bin"
	model="model.django_word128_encoder256_rule128_node64.beam15.adam.simple_trans.no_unary_closure.8e39832.run3.best_acc.npz"
	commandline="-rule_embed_dim 128 -node_embed_dim 64"
	datatype="django"
fi

# run interactive mode on trained models
THEANO_FLAGS="mode=FAST_RUN,device=${device},floatX=float32" python code_gen.py \
	-data_type ${datatype} \
	-data ${dataset} \
	-output_dir ${output} \
	-model models/${model} \
	${commandline} \
	interactive \
	-mode new