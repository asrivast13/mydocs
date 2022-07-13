#!/bin/bash 

ORIGAUDIO=../audio
WAVDIR=../wav
CONFIGDIR=../config
LOGDIR=../log

LABELFILE=./corpus.labels

## See end of this script for example of CSV file
if [ $# -eq 0 ]
then
    echo "No arguments supplied: Usage: $0 <CSV file> [<Output labels file>]"
    exit
fi

CSVFILE=$1

if [ $# -eq 2 ]
then
   LABELFILE=$2;
fi

[ ! -d $ORIGAUDIO ] && mkdir $ORIGAUDIO
[ ! -d $WAVDIR ] && mkdir $WAVDIR
[ ! -d $CONFIGDIR ] && mkdir $CONFIGDIR
[ ! -d $LOGDIR ] && mkdir $LOGDIR

#rm -rf $ORIGAUDIO/*
#rm -rf $WAVDIR/*
#rm -rf $CONFIGDIR/*
#rm -rf $LOGDIR/*

#rm -f $LABELFILE

re='^[01]$'

cat $CSVFILE | while read -r line || [[ -n $line ]];
	do 
		IFS=',' read -r -a array <<< "$line";
		if [[ ${#array[@]} -ne 3 ]]; then
			echo "Line $line in CSV file is malformed ... skipping";
			continue;
		fi

		if ! [[ ${array[2]} =~ $re ]]; then
			echo "Line $line in CSV file is malformed ... skipping";
			continue;
		fi
		id=${array[0]};
		target=${array[2]};

		if [ -e $ORIGAUDIO/$id.mp3 ]; then
			echo "Skipping $id which was already completed!";
			continue;
		fi

		wget -a $LOGDIR/config_download.log -O $CONFIGDIR/$id.json "https://po-call-recording.talkdeskapp.com/recording/$id";
		rc=$?; if [[ $rc != 0 ]]; then echo -e "Downloading config JSON for $id Failed! Skipping ..."; rm -f $CONFIGDIR/$id.json; continue; fi

		audiouri=$(cat $CONFIGDIR/$id.json | jq -r .url);
		AMDoutput=$(cat $CONFIGDIR/$id.json | jq -r .metadata.machine_result);
		amdlabel=1;
		if [[ $AMDoutput == "machine" ]]; then amdlabel=0; fi
		
		wget -a $LOGDIR/audio_download.log -O $ORIGAUDIO/$id.mp3 $audiouri;
		rc=$?; if [[ $rc != 0 ]]; then echo -e "Downloading audio MP3 for $id Failed! Skipping ..."; rm -f $ORIGAUDIO/$id.mp3; continue; fi
		dur=$(soxi -D $ORIGAUDIO/$id.mp3);

		sox $ORIGAUDIO/$id.mp3 -b 16 -e signed -r 8000 -L -c 1 $WAVDIR/$id.wav >> $LOGDIR/convert_audio.log 2>&1;
		#echo -e "$id \t $WAVDIR/$id.wav" >| $CONFIGDIR/$id.scp;
		
		echo -e "$id,$dur,$target,$amdlabel" >> $LABELFILE;

		echo -e "$id \t $dur \t $target \t $amdlabel";
	done

exit $rc
