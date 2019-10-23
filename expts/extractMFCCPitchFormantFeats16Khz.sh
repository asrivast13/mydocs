#!/bin/bash

scp=$1

[ ! -d "../kaldi" ] && mkdir "../kaldi"
[ ! -d "../featex" ] && mkdir "../featex"
[ ! -d "../merged" ] && mkdir "../merged"
[ ! -d "../logs" ] && mkdir "../logs"
[ ! -d "../scp" ] && mkdir "../scp"

counter=1
#rm -f ../kaldi/*
#rm -f ../featex/*
#rm -f ../merged/*
#rm -f ../logs/*
#rm -f ../scp/*

for id in `cat $scp | awk '{print $1}'`; 
	do 
	  audio=$(grep -w "$id" $scp | awk '{print $2}'); 
	  rc=$?; if [[ $rc != 0 ]]; then echo -e "$counter \t\t:\t $id \t:\t grep failed!"; fi 


	  if [[ $rc == 0 ]]; 
		then
	  		echo -e "$id \t $audio" >| ../scp/$id.scp
	  		~/projects/kaldi/src/featbin/compute-mfcc-feats --sample-frequency=16000 --use-energy=false --num-ceps=15 "scp:../scp/$id.scp" "ark,scp,t,f:../kaldi/$id.mfcc.ark,../kaldi/$id.mfcc.scp" >> ../logs/mfcc.log 2>&1;

	  		rc=$?; if [[ $rc != 0 ]]; then echo -e "$counter \t\t:\t $id \t:\t mfcc failed!"; fi 
		fi

	  if [[ $rc == 0 ]];
		then
	  		~/projects/kaldi/src/featbin/compute-and-process-kaldi-pitch-feats --simulate-first-pass-online=true --add-delta-pitch=false --frames-per-chunk=10 --sample-frequency=16000 "scp:../scp/$id.scp" "ark,scp,t,f:../kaldi/$id.pitch.ark,../kaldi/$id.pitch.scp" >> ../logs/pitch.log 2>&1;

	  		rc=$?; if [[ $rc != 0 ]]; then echo -e "$counter \t\t:\t $id \t:\t pitch failed!"; fi 
		fi

	  if [[ $rc == 0 ]];
		then
	  		featEx.babac91.exe -i $audio -u $id -o ../featex/$id.featx.ark >> ../logs/featEx.log 2>&1;

	  		rc=$?; if [[ $rc != 0 ]]; then echo -e "$counter \t\t:\t $id \t:\t featex failed!"; fi 
		fi

	  if [[ $rc == 0 ]];
		then
	  		~/projects/kaldi/src/featbin/paste-feats --length-tolerance=2 ark:../kaldi/$id.pitch.ark ark:../kaldi/$id.mfcc.ark ark:../featex/$id.featx.ark "ark,t:../merged/$id.feats.ark" &>> ../logs/paste.log; 

	  		rc=$?; if [[ $rc != 0 ]]; then echo -e "$counter \t\t:\t $id \t:\t paste failed!"; fi 
		fi

	  if [[ $rc == 0 ]];
		then
	  		nfeats=$(tail -n +2 ../merged/$id.feats.ark | wc -l);
	  		echo -e "$counter \t\t:\t $id \t:\t $nfeats";
		fi

	  counter=$((counter+1))
	done

