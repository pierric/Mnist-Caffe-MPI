DLLS :=  ${CAFFEDIR}/lib/libcaffe.so.1.0.0 \
		 ${OPENCVDIR}/lib/libopencv_core.so.3.2.0 \
         ${OPENCVDIR}/lib/libopencv_imgproc.so.3.2.0 \
		 ${OPENCVDIR}/lib/libopencv_imgcodecs.so.3.2.0 \
	     ${HDFDIR}/lib/libhdf5_hl.so.100.0.1 \
		 ${HDFDIR}/lib/libhdf5.so.101.0.0
all:
	$(foreach dll, ${DLLS}, cp ${dll} .;)
	ln -sf libopencv_core.so.3.2.0 libopencv_core.so.3.2
	ln -sf libopencv_imgproc.so.3.2.0 libopencv_imgproc.so.3.2
	ln -sf libopencv_imgcodecs.so.3.2.0 libopencv_imgcodecs.so.3.2
	ln -sf libhdf5_hl.so.100.0.1 libhdf5_hl.so.100
	ln -sf libhdf5.so.101.0.0 libhdf5.so.101