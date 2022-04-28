#!/bin/bash -ex

CREATEREF="true"
VADPOWER=2
WINLEN=20
CTCALPHA=1
CTCBETA=10
CTCBEAMW=100
VERBOSITY=2
USELM="true"
SEGMENT="true"

if [ $# -ne 2 ]
then
    echo "No arguments supplied: Usage: $0 [vosk|w2v2] <experiment name>"
    exit
fi

engine=$1
expName=$2

odir="../expts/$engine-$expName"
sttoutdir="$odir/sttoutput"
scoredir="$odir/score"

[ ! -d $odir ] && mkdir $odir
rm -rf $odir/*

[ ! -d $sttoutdir ] && mkdir $sttoutdir
[ ! -d $scoredir ]  && mkdir $scoredir


if [[ $CREATEREF == "true" ]];
then
[ ! -d "../ref" ] && mkdir ../ref
rm -rf *.stm*

for txt in /mnt/d/Data/Speech/MagicData/Spanish_Conversational_Speech_Test_Corpus/TXT/*.txt; do fb=$(basename $txt .txt); perl -ne 'BEGIN{use File::Basename; use File::Slurper 'read_text';  $filename = basename $ARGV[0], ".txt"; $header = read_text("stm_header.txt"); print "$header\n"; $prevEnd = 0.0;}{chomp; next if /^\s*$/; my @tokens = split /\t/, $_; $speaker = $tokens[1]; $gender=$tokens[2]; $text=lc(" " . $tokens[3]); $ignore = 0; $ignore = 1 if $text =~ /Beijing Magic Data/i; if($gender eq "none" or $speaker eq 0){ $speaker="unknown";$gender="unk";$ignore=1;} $tag="<O,"."F1".",".$gender.">"; $tokens[0]=~s/[\[\]]//g; my @times=split /\,/, $tokens[0]; $text =~ s/[\,\?]//g; $text=~s/ mmm\.+ / \(%hesitation\) /g; $text=~s/ [uhm|hmm] / \(%hesitation\) /g; if(($text=~/^[\W\s]*$/) || ($text=~/\+/)){$ignore=1;} $text="IGNORE_TIME_SEGMENT_IN_SCORING" if $ignore; if(($times[0]-$prevEnd)>3.0){printf("%s 1 Filler %s %s <O,FX,unk> IGNORE_TIME_SEGMENT_IN_SCORING\n", $filename, $prevEnd, $times[0]);} $prevEnd = $times[1]; printf("%s 1 %s %s %s %s %s\n", $filename, $speaker, $times[0], $times[1], $tag, $text);}' $txt >| ../ref/$fb.stm; echo $fb done; done
cat ../ref/*.stm >| ./magicdataEsConv.stm
fi

if [[ $engine == "w2v2" ]];
then
	if [[ $USELM == "true" ]];
	then
	python -u -W ignore ~/Source/mydocs/scripts/run_stt.py --model ~/Models/wav2vec2-large-xlsr-53-spanish/ --inpath /mnt/d/Data/Speech/MagicData/Spanish_Conversational_Speech_Test_Corpus/WAV/ --engine w2v2 --outpath "$sttoutdir" --vadpower $VADPOWER --winlen $WINLEN --ctcalpha $CTCALPHA --ctcbeta $CTCBETA --ctcbeamw $CTCBEAMW --uselm --verbosity $VERBOSITY;
	rc=$?; if [[ $rc != 0 ]]; then echo -e "Run W2V2 STT Failed!" && exit $rc; fi
	else
	python -u -W ignore ~/Source/mydocs/scripts/run_stt.py --model ~/Models/wav2vec2-large-xlsr-53-spanish/ --inpath /mnt/d/Data/Speech/MagicData/Spanish_Conversational_Speech_Test_Corpus/WAV/ --engine w2v2 --outpath "$sttoutdir" --vadpower $VADPOWER --winlen $WINLEN --verbosity $VERBOSITY;
	rc=$?; if [[ $rc != 0 ]]; then echo -e "Run W2V2 STT Failed!" && exit $rc; fi
	fi
else
	if [[ $SEGMENT == "true" ]];
	then
	python -u -W ignore ~/Source/mydocs/scripts/run_stt.py --model ~/Models/vosk-model-small-es-0.22 --inpath /mnt/d/Data/Speech/MagicData/Spanish_Conversational_Speech_Test_Corpus/WAV/ --engine vosk --outpath $sttoutdir --vadpower $VADPOWER --winlen $WINLEN --segment;
	rc=$?; if [[ $rc != 0 ]]; then echo -e "Run Vosk STT Failed!" && exit $rc; fi
	else
	python -u -W ignore ~/Source/mydocs/scripts/run_stt.py --model ~/Models/vosk-model-small-es-0.22 --inpath /mnt/d/Data/Speech/MagicData/Spanish_Conversational_Speech_Test_Corpus/WAV/ --engine vosk --outpath $sttoutdir;
	rc=$?; if [[ $rc != 0 ]]; then echo -e "Run Vosk STT Failed!" && exit $rc; fi
	fi
fi

cat $sttoutdir/*.ctm > $scoredir/$expName.ctm

hubscr.pl -v -g ../misc/spanish.glm -h hub4 -l spanish -F stm -r ./magicdataEsConv.stm -f ctm $scoredir/$expName.ctm

rc=$?;
if [[ $rc == 0 ]];
then
    grep "Sum/Avg" $scoredir/$expName.ctm.filt.sys
fi


exit $rc

##### stm_header.txt #############################################
###  ;; CATEGORY "0" "" ""
###  ;; LABEL "O" "Overall" "Overall"
###  ;;
###  ;; CATEGORY "1" "Hub4 Focus Conditions" ""
###  ;; LABEL "F0" "Baseline//Broadcast//Speech" ""
###  ;; LABEL "F1" "Spontaneous//Broadcast//Speech" ""
###  ;; LABEL "F2" "Speech Over//Telephone//Channels" ""
###  ;; LABEL "F3" "Speech in the//Presence of//Background Music" ""
###  ;; LABEL "F4" "Speech Under//Degraded//Acoustic Conditions" ""
###  ;; LABEL "F5" "Speech from//Non-Native//Speakers" ""
###  ;; LABEL "FX" "All other speech" ""
###  ;; CATEGORY "2" "Speaker Sex" ""
###  ;; LABEL "female" "Female" ""
###  ;; LABEL "male"   "Male" ""
###  ;; LABEL "child"   "Child" ""
###  ;; LABEL "unk"   "Unknown" ""
###
####
####
############# spanish.glm ########################################
###  ;;  file: spanish.glm
###  ;;  desc:  this file contains the mapping rules to be applied to both
###  ;;  reference and system output transcripts to be used in rt-04f stt 
###  ;;  this file contains additional glm entries added during the rt-03s
###  ;;  evaluation and was derived from en20030506.glm
###  ;;
###  * name "spanish.glm"
###  * desc "Empty French glm "
###  * format = 'NIST1'
###  * max_nrules = '2500'
###  * copy_no_hit = 'T'
###  * case_sensitive = 'F'
###  
###  
###  ;;  backchannel mappings
###  huh-uh  =>  %bcnack    / [ ] __ [ ] ;; negative token
###  huhuh   =>  %bcnack    / [ ] __ [ ] ;; negative token
###  uh-uh   =>  %bcnack    / [ ] __ [ ] ;; negative token
###  uhuh    =>  %bcnack    / [ ] __ [ ] ;; negative token
###  mhm     =>  %bcack     / [ ] __ [ ] ;; affirmative token 
###  mm-hm   =>  %bcack     / [ ] __ [ ] ;; affirmative token 
###  mm-hmm  =>  %bcack     / [ ] __ [ ] ;; affirmative token 
###  mm-huh  =>  %bcack     / [ ] __ [ ] ;; affirmative token 
###  mmhm    =>  %bcack     / [ ] __ [ ] ;; affirmative token 
###  uh-hmm  =>  %bcack     / [ ] __ [ ] ;; affirmative token 
###  uhhuh   =>  %bcack     / [ ] __ [ ] ;; affirmative token
###  uh-huh  =>  %bcack     / [ ] __ [ ] ;; affirmative token
###  um-hmm  =>  %bcack     / [ ] __ [ ] ;; affirmative token 
###  um-hum  =>  %bcack     / [ ] __ [ ] ;; affirmative token 
###  
###  
###  ;; hesitation sound mappings
###  ;; modified by viet-bac le: for now, we decide to remove all hesitations from the reference
###  ach     =>       / [ ] __ [ ] ;; hesitation token
###  ah      =>       / [ ] __ [ ] ;; hesitation token 
###  eee     =>       / [ ] __ [ ] ;; hesitation token 
###  eh      =>       / [ ] __ [ ] ;; hesitation token 
###  er      =>       / [ ] __ [ ] ;; hesitation token 
###  ew      =>       / [ ] __ [ ] ;; hesitation token 
###  ha      =>       / [ ] __ [ ] ;; hesitation token 
###  hee     =>       / [ ] __ [ ] ;; hesitation token 
###  hm      =>       / [ ] __ [ ] ;; hesitation token 
###  hmm     =>       / [ ] __ [ ] ;; hesitation token (pawel)
###  huh     =>       / [ ] __ [ ] ;; hesitation token 
###  mm      =>       / [ ] __ [ ] ;; hesitation token 
###  oof     =>       / [ ] __ [ ] ;; hesitation token 
###  uh      =>       / [ ] __ [ ] ;; hesitation token 
###  um      =>       / [ ] __ [ ] ;; hesitation token 
###  ;;
###  %ach     =>  %hesitation     / [ ] __ [ ] ;; hesitation token 
###  %ah      =>  %hesitation     / [ ] __ [ ] ;; hesitation token 
###  %eee     =>  %hesitation     / [ ] __ [ ] ;; hesitation token 
###  %eh      =>  %hesitation     / [ ] __ [ ] ;; hesitation token 
###  %er      =>  %hesitation     / [ ] __ [ ] ;; hesitation token 
###  %ew      =>  %hesitation     / [ ] __ [ ] ;; hesitation token 
###  %ha      =>  %hesitation     / [ ] __ [ ] ;; hesitation token 
###  %hee     =>  %hesitation     / [ ] __ [ ] ;; hesitation token 
###  %hm      =>  %hesitation     / [ ] __ [ ] ;; hesitation token 
###  %huh     =>  %hesitation     / [ ] __ [ ] ;; hesitation token 
###  %mm      =>  %hesitation     / [ ] __ [ ] ;; hesitation token 
###  %oof     =>  %hesitation     / [ ] __ [ ] ;; hesitation token 
###  %uh      =>  %hesitation     / [ ] __ [ ] ;; hesitation token 
###  %um      =>  %hesitation     / [ ] __ [ ] ;; hesitation token 
###  %hmm	 =>  %hesitation     / [ ] __ [ ] ;; hesitation toke (pawel)
###  
