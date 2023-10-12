docker run -it \
	-v `pwd`:/kubric \
	kubricdockerhub/kubruntu python /kubric/syndata_generator/tfrecord_creator.py \
	--input-dir syndata_generator/random_texture \
	--output-folder tfrecord_output_test

