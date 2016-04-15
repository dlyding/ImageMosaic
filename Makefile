ALGNAMES := Image DrawCircle Template Harris SomeFunction LBPDescriptor CovarianceMatrix RansacForImage
			
INCFILES := ErrorCode.h $(addsuffix .h, $(ALGNAMES))
OBJFILES := main.o $(addsuffix .o, $(ALGNAMES))

EXEFILE  := okanoexec

NVCCCMD  := nvcc
NVCCFLAG := -arch=sm_20
NVLDFLAG := -lnppi

world: $(EXEFILE)

$(EXEFILE): $(OBJFILES)
	$(NVCCCMD) $(OBJFILES) -o $(EXEFILE) $(NVLDFLAG)

$(OBJFILES): %.o:%.cu $(INCFILES)
	$(NVCCCMD) -c $(filter %.cu, $<) -o $@ $(NVCCFLAG)

clean:
	rm -rf $(OBJFILES) $(EXEFILE)
