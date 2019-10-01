bin_PROGRAMS=aepd
aepd_SOURCES=main.c websocket-protocol.c
aepd_CFLAGS=-fPIC -Wall -Werror -D_FORTIFY_SOURCE=2 -fstack-protector -std=gnu99 -pedantic -DINSTALL_DATADIR=\"${datarootdir}\"
aepd_LDFLAGS=-fPIC
aepd_LDADD=-L../libarmep -larmep -lwebsockets

#
# cook a random test cert and key
# notice your real cert and key will want to be 0600 permissions
#
aepd.pem aepd.key.pem:
	printf "GB\nErewhon\nAll around\naepd-test-cert\n\nlocalhost\nnone@invalid.org\n" | \
	openssl req -new -newkey rsa:1024 -days 10000 -nodes -x509 -keyout \
	./aepd.key.pem -out ./aepd.pem >/dev/null 2>&1  && \
	chmod 644 	./aepd.key.pem \
			./aepd.pem

clean-local:
	rm -f ./aepd.key.pem ./aepd.pem

install-data-local: aepd.pem aepd.key.pem
	mkdir -p $(DESTDIR)$(datadir)/aepd
	cp share/aepscope.html \
	   share/favicon.ico \
	   aepd.pem \
	   aepd.key.pem \
	   share/linaro-logo-32.png \
	   share/caliper.png \
	   share/trigger.png \
	   share/indicator-bg.png \
	   share/indicator-triggered.png \
		$(DESTDIR)$(datadir)/aepd

