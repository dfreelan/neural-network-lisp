TMPFILE1=./tmp1.$$
TMPFILE2=./tmp2.$$
cr="
"
for filename in *.txt; do
  tr ' ' '\n'  < $filename > $TMPFILE1 
  tr -s '\n'  < $TMPFILE1 > $TMPFILE2
  #awk '{gsub(/\\n\\n/,"\n")}1' $filename > $TMPFILE2
  mv $TMPFILE2 $filename # Comment this line out to not overwrite the original file.

done
