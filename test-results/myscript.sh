TMPFILE=./tmp.$$
cr="
"
for filename in *.txt; do
  tr -s '\n'  < $filename > $TMPFILE
  #awk '{gsub(/\\n\\n/,"\n")}1' $filename > $TMPFILE
  mv $TMPFILE $filename # Comment this line out to not overwrite the original file.
done
