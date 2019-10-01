lib_LTLIBRARIES=libarmep.la
include_HEADERS=libarmep.h aepd-interface.h
dist_libarmep_la_SOURCES=sample.c \
				interpolation.c \
				configuration.c \
				avg-mean-us.c \
				protocol.c \
				service.c \
				aepd-interface.c \
				libarmep.h \
				aepd-interface.h

libarmep_la_CFLAGS=-Wall -Werror -std=gnu99 -pedantic -pthread -D_FORTIFY_SOURCE=2 -fstack-protector
libarmep_la_LDFLAGS=-pthread

libarmep_la_CFLAGS+= -c \
	-DINSTALL_DATADIR=\"@datadir@\" 
libarmep_la_LDFLAGS+= 

