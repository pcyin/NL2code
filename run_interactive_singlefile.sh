device="cpu"

if [ "$1" == "hs" ]; then
	# hs dataset
	echo "run trained model for hs"
	dataset="data/hs.freq3.pre_suf.unary_closure.bin"
	model="models/model.hs_unary_closure_top20_word128_encoder256_rule128_node64.beam15.adadelta.simple_trans.8e39832.iter5600.npz"
	datatype="hs"
else
	# django dataset
	echo "run trained model for django"
	dataset="data/django.cleaned.dataset.freq5.par_info.refact.space_only.bin"
	model="models/model.django_word128_encoder256_rule128_node64.beam15.adam.simple_trans.no_unary_closure.8e39832.run3.best_acc.npz"
	datatype="django"
fi

# run interactive mode on trained models
# run interactive mode on trained models
THEANO_FLAGS="mode=FAST_RUN,device=${device},floatX=float32" python interactive_mode.py \
	-data_type ${datatype} \
	-data ${dataset} \
	-model ${model}