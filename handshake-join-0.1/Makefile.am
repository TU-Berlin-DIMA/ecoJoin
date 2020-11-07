
bin_PROGRAMS = sj numainfo joinperf

sj_SOURCES = comm/ringbuffer.c \
	     comm/ringbuffer.h \
	     worker/worker.c \
	     worker/worker.h \
	     messages/core2core.h \
	     messages/result.h \
	     time/time.h \
	     time/time.c \
	     data/data.h \
	     data/generate.c \
	     mem/mem.h \
	     mem/mem.c \
	     debug/debug.h \
	     parameters.h \
	     main.c

numainfo_SOURCES = numainfo.c

joinperf_SOURCES = master/master.h \
		   data/data.h \
		   data/generate.c \
		   mem/mem.h \
		   mem/mem.c \
		   joinperf.c

EXTRA_DIST = tools/iterate-cores.pl \
             tools/find-max-rate.pl \
             doxygen.conf \
	     README
