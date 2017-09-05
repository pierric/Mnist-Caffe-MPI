CAFFEDIR    := /home/jwu/jp/caffe/cppbuild/linux-x86_64
OPENCVDIR   := /home/jwu/jp/opencv/cppbuild/linux-x86_64
HDFDIR      := /home/jwu/jp/hdf5/cppbuild/linux-x86_64
OPENBLASDIR := /home/jwu/jp/openblas/cppbuild/linux-x86_64
INCDIR      := ${CAFFEDIR}/include
LINKOPTS    := -pthread -L${CAFFEDIR}/lib -L${OPENCVDIR}/lib -L${HDFDIR}/lib -lcaffe -lglog -lgflags -lleveldb -llmdb -lopenblas -lprotobuf -lsnappy -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lz -lpthread -lboost_filesystem -lboost_system -lboost_thread -lboost_mpi -lboost_serialization -Wl,-rpath=./libs
LDPATH      := ${CAFFEDIR}/lib:${OPENCVDIR}/lib:${HDFDIR}/lib:${OPENBLASDIR}/lib
REMOTEHOSTS := dev004 dev005 dev007
export CAFFEDIR OPENCVDIR HDFDIR

.PHONY: libs deploy
main: main.cpp
	mpiCC -O3 -o $@ -DCPU_ONLY -I${INCDIR} -std=gnu++11 -Wno-deprecated-declarations $< ${LINKOPTS}
libs:
	$(MAKE) -C libs all
deploy: main libs
	$(foreach host, ${REMOTEHOSTS}, rsync main ${host}:/home/jwu/Mnist-MPI;)
	$(foreach host, ${REMOTEHOSTS}, rsync -drlv model ${host}:/home/jwu/Mnist-MPI;)
	$(foreach host, ${REMOTEHOSTS}, rsync -drlv libs ${host}:/home/jwu/Mnist-MPI;)

run:
	mpirun -x LD_LIBRARY_PATH=./libs --report-bindings --hostfile hostfile1 -rf rankfile1 ./main
