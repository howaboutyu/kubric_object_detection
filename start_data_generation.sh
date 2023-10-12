docker run -it \
	-v `pwd`:/kubric \
	kubricdockerhub/kubruntu python /kubric/syndata_generator/kubric_generator.py \
	--texture-dir syndata_generator/generated_textures  \
	--output-path kubric_synthetic_data_output \
	--num-generation 2000

