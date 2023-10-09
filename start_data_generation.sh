git clone  https://github.com/google-research/kubric.git 
cd kubric && git checkout a749598fd84f7265c130dd41f37ebcbc6bba0c65 && cd ..
docker run -it \
	-v `pwd`:/kubric \
	kubricdockerhub/kubruntu bash
