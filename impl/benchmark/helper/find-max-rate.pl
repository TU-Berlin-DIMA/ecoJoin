#!/usr/bin/perl -w
#
# Find maximum throughput rate that handshake join can sustain;
# uses the bisection method to find this rate.
#
# Author: Jens Teubner <jens.teubner@inf.ethz.ch>
#
# (c) 2010 ETH Zurich, Systems Group
#
# $Id: find-max-rate.pl 603 2010-08-17 07:35:46Z jteubner $
#

use strict;

use Getopt::Long;

my $min           = 100.;
my $max           = 10000.;
my $cores         = 2;
my $duration      = 3600;
my $window_size_R = 10 * 60;
my $window_size_S = 10 * 60;
my $num_tuples;
my $rate;

Getopt::Long::Configure ("bundling");

my $opts = GetOptions ('min=i'             => \$min,
                       'max=i'             => \$max,
                       'cores|c=i'         => \$cores,
                       'duration|d=i'      => \$duration,
                       'window-size-r|w=i' => \$window_size_R,
                       'window-size-s|W=i' => \$window_size_S);

$rate = ($min + $max) / 2;
#while ($max - $min >= 2)
while (($max - $min) / ($max + $min) >= 0.005)
{
    my $num_tuples = $duration * $rate;

    my $retval = system ("../../bin/gpu_stream -r $rate -R $rate -n $num_tuples -N $num_tuples -w $window_size_R -W $window_size_S -p ht_cpu4 -l");

    print "Run with rate $rate " . ($retval == 0 ? "was successful" : "failed");
    print ".\n";

    if ($retval == 0)
    {
        $min = $rate;
    }
    else
    {
        $max = $rate;
    }

    $rate = ($min + $max) / 2;
}

print "Rate is $rate.\n";
