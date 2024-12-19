docker run -d \
	--gpus all \
	--ipc=host \
	--privileged \
	--rm -it \
	--user user \
	--name cuda \
	-e "CUDA_VISIBLE_DEVICES=0" \
	-e "CUDA_DEVICE_ORDER=PCI_BUS_ID" \
	-v $PWD:/home/user/ML_cuda \
	clipdino-torch20-cu115-triton3
