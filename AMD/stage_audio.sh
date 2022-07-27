#!/bin/bash

SEGSIZE=2 # seconds

if [ $# -lt 2 ]
  then
    echo "No arguments supplied: Usage: $0 <Stats file> <audio folder> [<output dir>]"
    exit
fi

stats=$1
audiodir=$2
odir=$audiodir
if [ $# -eq 3 ]
  then
    odir=$3;
  else
    dset=$(basename $stats ".stats");
    odir="$audiodir/$dset";
fi

[ ! -d "$odir" ] && mkdir "$odir"

cat $stats | while read -r line || [[ -n $line ]];
        do 
          id=$(echo $line | awk '{print $1}');
          ref=$(echo $line | awk '{print $3}');
	  class="vm"; 
	  if [[ $ref == 1 ]]; 
	  then 
	    class="lp"; 
	  fi;

	  gen='u'
	  fb=$(echo $class"_"$gen"_"$id);
          sox -V $audiodir/$id.mp3 $odir/$fb.flac trim 0 $SEGSIZE;
	done
 
