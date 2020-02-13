#!/usr/bin/perl -w
#
# Iterate a list of core counts, then invoke find-max-rate.pl
# for each core count to determine the maximum throughput for
# this degree of parallelism.
#
# Author: Jens Teubner <jens.teubner@inf.ethz.ch>
#
# (c) 2010 ETH Zurich, Systems Group
#
# $Id: iterate-cores.pl 607 2010-08-17 09:40:08Z jteubner $
#

use strict;

use Getopt::Long;

use IO::Handle;

my $min           = 100.;
my $max           = 10000.;
my $duration      = 3600;
my $window_size_R = 10 * 60;
my $window_size_S = 10 * 60;

my $rate          = undef;
my $cores         = undef;
my $cores_last    = undef;
my $rate_last     = undef;

my %rates;

my $uname = `uname -a`;
my $os;

if ($uname =~ /Linux/)
{
    $os = "Linux";
}
elsif ($uname =~ /Darwin/)
{
    $os = "MacOS";
}
else
{
    die "Unknown operating system";
}

Getopt::Long::Configure ("bundling");

my $opts = GetOptions ( 'min=i'             => \$min,
                        'max=i'             => \$max,
                        'duration|d=i'      => \$duration,
                        'window-size-r|w=i' => \$window_size_R,
                        'window-size-s|W=i' => \$window_size_S );

while ($cores = shift)
{

    # Estimate performance (based on previous measurement).
    if (defined $cores_last and defined $rate_last)
    {
        my $estimated_rate = $rate_last * sqrt ($cores / $cores_last);

        $min = int($estimated_rate / sqrt(2));
        $max = int($estimated_rate * sqrt(2));
    }

    # Call find-max-rate.pl
    my $cmdline;
    if ($os eq 'Linux')
    {
        $cmdline = "script -a -f -c './find-max-rate.pl -c $cores --min $min --max $max -w $window_size_R -W $window_size_S -d $duration' iterate.$$.$cores-cores.log";
    }
    else
    {
        $cmdline = "script -a iterate.$$.$cores-cores.log './find-max-rate.pl -c $cores --min $min --max $max -w $window_size_R -W $window_size_S -d $duration'";
    }
    #my $cmdline = "./find-max-rate.pl -c $cores --min $min --max $max -w $window_size_R -W $window_size_S -d $duration 2>&1 | tee -a iterate.$$.$cores-cores.log";

    open (LOG, "> iterate.$$.$cores-cores.log") || die "Error opening log: $!";
    print LOG $cmdline;
    close LOG;

    print STDOUT "Calling `$cmdline'.\n";

    system $cmdline;

    open (LOG, "< iterate.$$.$cores-cores.log");
    my @output = <LOG>;
    close LOG;

    # find-max-rate.pl now should have computed a rate.
    foreach (@output)
    {
        if ($_ =~ /Rate is ([0-9]+\.?[0-9]*)\./)
        {
            $rates{$cores} = $1;
        }
    }

    if (defined $rates{$cores})
    {
        print STDOUT "-- Rate returned by find-max-rate.pl for $cores cores was " . $rates{$cores} . ".\n";

        $cores_last = $cores;
        $rate_last  = $rates{$cores};
    }
    else
    {
        print STDERR "WARNING: No rate was computed.\n";
    }

}

print "\nSUMMARY:\n";
foreach (keys %rates)
{
    print $_ . "\t" . $rates{$_} . "\n";
}
