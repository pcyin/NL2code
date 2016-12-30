Neural Code Generation with Syntax Guidance

## Training 

```
dataset="django.cleaned.dataset.freq5.par_info.refact.space_only.bin"
commandline="-batch_size 10 -max_epoch 50 -valid_per_batch 4000 -save_per_batch 4000 -decode_max_time_step 100 -optimizer adam -rule_embed_dim 128 -node_embed_dim 64 -train_patience 7 -valid_metric accuracy -no_parent_hidden_state_feed -no_parent_action_feed"
datatype="django"

THEANO_FLAGS="mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=0.3" python -u code_gen.py \
	-data_type ${datatype} \
	-data ../data/${dataset} \
	-output_dir ${output} \
	${commandline} \
	train
```
