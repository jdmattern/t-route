# compiler
FC := gfortran

# compile flags
FCFLAGS = -fbounds-check -g -c -O2 -fPIC

arrays_module.o: arrays_module.f90
	$(FC) $(FCFLAGS) -o $@ $<
    
matrix_module.o: matrix_module.f90
	$(FC) $(FCFLAGS) -o $@ $<
    
var_module.o: var_module.f90
	$(FC) $(FCFLAGS) -o $@ $<

arrays_section_module.o: arrays_section_module.f90
	$(FC) $(FCFLAGS) -o $@ $<

xsec_attribute_module.o: xsec_attribute_module.f90
	$(FC) $(FCFLAGS) -o $@ $<

constants_module.o: constants_module.f90
	$(FC) $(FCFLAGS) -o $@ $<

subtools.o: subtools.f90
	$(FC) $(FCFLAGS) -o $@ $<

moussa_network.o: moussa_network.f90
	$(FC) $(FCFLAGS) -o $@ $<

diffusive_cnt.o: diffusive_cnt.f90
	$(FC) $(FCFLAGS) -o $@ $<

pydiffusive_cnt.o: pydiffusive_cnt.f90
	$(FC) $(FCFLAGS) -o $@ $<

install:
	cp *.o ../../../../src/python_routing_v02/troute/routing/fast_reach
	cp *.mod ../../../../src/python_routing_v02/troute/routing/fast_reach   
clean:
	rm -f *.o *.mod
