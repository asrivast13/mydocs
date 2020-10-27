#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "No arguments supplied: Usage: $0 <SCP file> [<output dir>]"
    exit
fi

scp=$1
odir="."
if [ $# -eq 2 ]
  then
    odir=$2;
  else
    dset=$(basename $scp ".scp");
    odir="../$dset";
fi
 
echo $odir;
exit;

[ ! -d "$odir/wav" ] && mkdir "$odir/wav"
[ ! -d "$odir/kaldi" ] && mkdir "$odir/kaldi"
[ ! -d "$odir/featex" ] && mkdir "$odir/featex"
[ ! -d "$odir/merged" ] && mkdir "$odir/merged"
[ ! -d "$odir/logs" ] && mkdir "$odir/logs"
[ ! -d "$odir/scp" ] && mkdir "$odir/scp"

counter=1
rm -f $odir/kaldi/*
rm -f $odir/featex/*
rm -f $odir/merged/*
rm -f $odir/logs/*
rm -f $odir/scp/*
rm -f $odir/wav/*

date

cat $scp | while read -r line || [[ -n $line ]];
	do 
	  id=$(echo $line | awk '{print $1}');
	  flac=$(echo $line | awk '{print $2}');
	  audio="$odir/wav/$id.wav"
	  #echo "$id | $flac | $audio"
	  sox $flac -b 16 -e signed -r 16000 -L -c 1 $audio >> $odir/logs/convert.log 2>&1;
	  rc=$?; if [[ $rc != 0 ]]; then echo -e "$counter \t\t:\t $id \t:\t sox failed!"; fi 
          echo -e "$id \t $audio" >| $odir/scp/$id.scp

	  if [[ $rc == 0 ]]; 
		then
	  		~/projects/kaldi/src/featbin/compute-fbank-feats --config=./fbank.conf "scp:$odir/scp/$id.scp" "ark,scp,t,f:$odir/kaldi/$id.fbank.ark,$odir/kaldi/$id.fbank.scp" >> $odir/logs/fbank.log 2>&1;

	  		rc=$?; if [[ $rc != 0 ]]; then echo -e "$counter \t\t:\t $id \t:\t fbank failed!"; fi 
		fi

	  if [[ $rc == 0 ]];
		then
	  		~/projects/kaldi/src/featbin/compute-and-process-kaldi-pitch-feats --simulate-first-pass-online=true --add-delta-pitch=false --frames-per-chunk=10 --sample-frequency=16000 "scp:$odir/scp/$id.scp" "ark,scp,t,f:$odir/kaldi/$id.pitch.ark,$odir/kaldi/$id.pitch.scp" >> $odir/logs/pitch.log 2>&1;

	  		rc=$?; if [[ $rc != 0 ]]; then echo -e "$counter \t\t:\t $id \t:\t pitch failed!"; fi 
		fi

	  if [[ $rc == 0 ]];
		then
	  		featEx -i $audio -u $id -o $odir/featex/$id.featx.ark >> $odir/logs/featEx.log 2>&1;

	  		rc=$?; if [[ $rc != 0 ]]; then echo -e "$counter \t\t:\t $id \t:\t featex failed!"; fi 
		fi

	  if [[ $rc == 0 ]];
		then
	  		~/projects/kaldi/src/featbin/paste-feats --length-tolerance=2 ark:$odir/kaldi/$id.pitch.ark ark:$odir/kaldi/$id.fbank.ark ark:$odir/featex/$id.featx.ark "ark,t:$odir/merged/$id.feats.ark" &>> $odir/logs/paste.log; 

	  		rc=$?; if [[ $rc != 0 ]]; then echo -e "$counter \t\t:\t $id \t:\t paste failed!"; fi 
		fi

	  if [[ $rc == 0 ]];
		then
	  		nfeats=$(tail -n +2 $odir/merged/$id.feats.ark | wc -l);
	  		echo -e "$counter \t\t:\t $id \t:\t $nfeats";
		fi

	  counter=$((counter+1))
	done

echo "All Done"
date

######## fbank.conf file used in this script ###################
### No non-default options for now.
##--window-type=hamming # disable Dans window, use the standard
##--use-energy=true    # only fbank outputs
##--sample-frequency=16000 # Cantonese is sampled at 8kHz
##
##--low-freq=64         # typical setup from Frantisek Grezl
##--high-freq=8000
##--dither=0
##--energy-floor=1.0
##
##--num-mel-bins=40     # 8kHz so we use 15 bins
##--htk-compat=false     # try to make it compatible with HTK
#################################################################

####### scp input to this script ###############################

##de_f_0809fd0642232f8c85b0b3d545dc2b5a.fragment1 	 /home/amisriv/audio/KaggleSpokenLID/orig/train/flac/de_f_0809fd0642232f8c85b0b3d545dc2b5a.fragment1.flac
##de_f_0809fd0642232f8c85b0b3d545dc2b5a.fragment1.noise1 	 /home/amisriv/audio/KaggleSpokenLID/orig/train/flac/de_f_0809fd0642232f8c85b0b3d545dc2b5a.fragment1.noise1.flac
##de_f_0809fd0642232f8c85b0b3d545dc2b5a.fragment1.noise10 	 /home/amisriv/audio/KaggleSpokenLID/orig/train/flac/de_f_0809fd0642232f8c85b0b3d545dc2b5a.fragment1.noise10.flac
##de_f_0809fd0642232f8c85b0b3d545dc2b5a.fragment1.pitch6 	 /home/amisriv/audio/KaggleSpokenLID/orig/train/flac/de_f_0809fd0642232f8c85b0b3d545dc2b5a.fragment1.pitch6.flac
##de_f_0809fd0642232f8c85b0b3d545dc2b5a.fragment1.speed8 	 /home/amisriv/audio/KaggleSpokenLID/orig/train/flac/de_f_0809fd0642232f8c85b0b3d545dc2b5a.fragment1.speed8.flac
##de_f_0809fd0642232f8c85b0b3d545dc2b5a.fragment10 	 /home/amisriv/audio/KaggleSpokenLID/orig/train/flac/de_f_0809fd0642232f8c85b0b3d545dc2b5a.fragment10.flac
###################################################################
