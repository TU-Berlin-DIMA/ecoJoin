#!/bin/bash
#
# Backup firmware.bin from the mountpoint before using...
#
# hold down the switch on the AEP and plug into USB
# CRP_DISABLED Mass storage device will appear containing one (fake) file
# called "firmware.bin".
#
# This utility lets you change the USB serial number in your AEP, which is
# necessary if you will use more than one.
#
# Example usage
#
# ./set-aep-serial.sh /run/media/agreen/CRP\ DISABLD/ 00010002

M=$1

if [ -e "$M/firmware.bin" ] ; then
 if [ ! -e ./original-firmware.bin ] ; then
  cp "$M/firmware.bin" ./original-firmware.bin
 fi
else
 echo "First arg need to be mountpoint"
 exit 1
fi

S=$2
if [ ${#S} -ne 8 ] ; then
	echo "Second arg is Serial number, needs to be 8 digits"
	exit 1
fi

cat ./original-firmware.bin | \
sed "s|\x1a\x03S\x00/\x00N\x00O\x00.\x00.\x00.\x00.\x00.\x00.\x00.\x00.\x00|\x1a\x03S\x00/\x00N\x00O\x00${S:0:1}\x00${S:1:1}\x00${S:2:1}\x00${S:3:1}\x00${S:4:1}\x00${S:5:1}\x00${S:6:1}\x00${S:7:1}\x00|g" | \
dd conv=nocreat,notrunc of="$M/firmware.bin"


