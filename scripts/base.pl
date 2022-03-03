#!/usr/bin/perl -w

use File::Basename;
use Getopt::Std;
$| = 1;

getopts('s:');

while(<>) {
   if($opt_s) {
	  $_ = basename $_, $opt_s;
  }
  else {
	 $_ = basename $_;
 } 
  print $_;
}

