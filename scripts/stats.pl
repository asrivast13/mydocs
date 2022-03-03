#!/usr/local/bin/perl

while(<>) {
        chop;
        if (!$min) {
                $min = $_;
        } elsif ($min > $_) {
                $min = $_;
        }

        if (!$max) {
                $max = $_;
        } elsif ($max < $_) {
                $max = $_;
        }

        $sum += $_;
        $sqr_sum += ($_*$_);
        $num++;
	push(@get_med, $_);
}

@sorted = sort numerically @get_med;
$med_idx = int($num/2);

print "\n";
print "Number of Data   :       $num\n";
print "--------------------------------\n";
print "Minimum          :       $min\n";
print "Maximum          :       $max\n";
print "Summation        :       $sum\n";
print "Average          :       ",$sum/$num,"\n";
print "Median           :       ",$sorted[$med_idx],"\n";
print "Standard Dev     :       ",sqrt($sqr_sum/$num-($sum/$num)*($sum/$num)),"\n";


sub numerically { $a <=> $b;}
