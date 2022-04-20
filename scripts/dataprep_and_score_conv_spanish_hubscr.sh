perl -ne 'BEGIN{use File::Basename; use File::Slurper 'read_text';  $filename = basename $ARGV[0], ".txt"; $header = read_text("stm_header.txt"); print "$header\n"; $prevEnd = 0.0;}{chomp; next if /^\s*$/; my @tokens = split /\t/, $_; $speaker = $tokens[1]; $gender=$tokens[2]; $text=lc(" " . $tokens[3]); $ignore = 0; $ignore = 1 if $text =~ /Beijing Magic Data/i; if($gender eq "none" or $speaker eq 0){ $speaker="unknown";$gender="unk";$ignore=1;} $tag="<O,"."F1".",".$gender.">"; $tokens[0]=~s/[\[\]]//g; my @times=split /\,/, $tokens[0]; $text =~ s/[\,\?]//g; $text=~s/ mmm\.+ / \(%hesitation\) /g; $text=~s/ [uhm|hmm] / \(%hesitation\) /g; if($text=~/^[\W\s]*$/){$ignore=1;} $text="IGNORE_TIME_SEGMENT_IN_SCORING" if $ignore; if(($times[0]-$prevEnd)>3.0){printf("%s 1 Filler %s %s <O,FX,unk> IGNORE_TIME_SEGMENT_IN_SCORING\n", $filename, $prevEnd, $times[0]);} $prevEnd = $times[1]; printf("%s 1 %s %s %s %s %s\n", $filename, $speaker, $times[0], $times[1], $tag, $text);}' A0001_S003_0_G0001_G0002.txt >| A0001_S003_0_G0001_G0002.stm

hubscr.pl -v -g spanish.glm -h hub4 -l spanish -F stm -r A0001_S003_0_G0001_G0002.stm -f ctm A0001_S003_0_G0001_G0002.ctm

# cat stm_header.txt
#;; CATEGORY "0" "" ""
#;; LABEL "O" "Overall" "Overall"
#;;
#;; CATEGORY "1" "Hub4 Focus Conditions" ""
#;; LABEL "F0" "Baseline//Broadcast//Speech" ""
#;; LABEL "F1" "Spontaneous//Broadcast//Speech" ""
#;; LABEL "F2" "Speech Over//Telephone//Channels" ""
#;; LABEL "F3" "Speech in the//Presence of//Background Music" ""
#;; LABEL "F4" "Speech Under//Degraded//Acoustic Conditions" ""
#;; LABEL "F5" "Speech from//Non-Native//Speakers" ""
#;; LABEL "FX" "All other speech" ""
#;; CATEGORY "2" "Speaker Sex" ""
#;; LABEL "female" "Female" ""
#;; LABEL "male"   "Male" ""
#;; LABEL "child"   "Child" ""
#;; LABEL "unk"   "Unknown" ""
