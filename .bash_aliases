#-------------------
# Personnal Aliases
#-------------------
alias gits='git status'
alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'
# -> Prevents accidentally clobbering files.
alias mkdir='mkdir -p'
alias h='history'
alias j='jobs -l'
alias which='type -a'
alias ..='cd ..'
alias ...='cd ../../../'
alias .4='cd ../../../../'
alias .5='cd ../../../../..'
alias update='sudo apt-get update'
alias bc='bc -l'
alias mount='mount |column -t'
alias now='date +"%T"'
alias nowtime=now
alias nowdate='date +"%d-%m-%Y"'
alias fastping='ping -c 100 -s.2'

## pass options to free ##
alias meminfo='free -m -l -t'
 
## get top process eating memory
alias psmem='ps auxf | sort -nr -k 4'
alias psmem10='ps auxf | sort -nr -k 4 | head -10'
 
## get top process eating cpu ##
alias pscpu='ps auxf | sort -nr -k 3'
alias pscpu10='ps auxf | sort -nr -k 3 | head -10'
 
## Get server cpu info ##
alias cpuinfo='lscpu'
 
## older system use /proc/cpuinfo ##
##alias cpuinfo='less /proc/cpuinfo' ##
 
## get GPU ram on desktop / laptop##
alias gpumeminfo='grep -i --color memory /var/log/Xorg.0.log'

# Pretty-print of some PATH variables:
alias path='echo -e ${PATH//:/\\n}'
alias libpath='echo -e ${LD_LIBRARY_PATH//:/\\n}'

alias du='du -kh'    # Makes a more readable output.
alias df='df -kTh'

alias cls=clear
alias clc=clear
alias re='source ~/.bashrc'

# Add colors for filetype and  human-readable sizes by default on 'ls':
alias ls='ls -h --color'
alias lx='ls -lXB'         #  Sort by extension.
alias lk='ls -lSr'         #  Sort by size, biggest last.
alias lt='ls -ltr'         #  Sort by date, most recent last.
alias lc='ls -ltcr'        #  Sort by/show change time,most recent last.
alias lu='ls -ltur'        #  Sort by/show access time,most recent last.
# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# The ubiquitous 'll': directories first, with alphanumeric sorting:
alias ll="ls -lv --group-directories-first"
alias lm='ll |more'        #  Pipe through 'more'
alias lr='ll -R'           #  Recursive ls.
alias la='ll -A'           #  Show hidden files.
alias tree='tree -Csuh'    #  Nice alternative to 'recursive ls' ...

#-------------------------------------------------------------
# Tailoring 'less'
#-------------------------------------------------------------
alias more='less'
export PAGER=less
export LESSCHARSET='utf-8'

#-------------------------------------------------------------
# Spelling typos - highly personnal and keyboard-dependent :-)
#-------------------------------------------------------------
alias xs='cd'
alias vf='cd'
alias moer='more'
alias moew='more'
alias kk='ll'

#------------------------------------------------------------
# Python-specific or Anaconda
#------------------------------------------------------------
alias python=python3
alias pip=pip3
alias p='python -u -W ignore '

## Top-10 aliases from open-source
# unpack a .tar file
alias untar='tar -zxvf '
# Want to download something but be able to resume if something goes wrong?
alias wget='wget -c '
# Need to generate a random, 20-character password for a new online account
alias getpass="openssl rand -base64 20"
# Downloaded a file and need to test the checksum
alias sha='shasum -a 256 '
# limit that to just five pings
alias ping='ping -c 5'
# Start a web server in any folder you'd like
alias www='python -m SimpleHTTPServer 8000'
# needed to know your external IP address
alias ipe='curl ipinfo.io/ip'
# Need to know your local IP address
alias ipi='ifconfig getifaddr eth0'
# clear the screen
alias c='clear'
# rsync between computers
alias rs='rsync -ravz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress'
