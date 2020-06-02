for i in {0..1}
do
   echo "bleh"
   python ./FL_simulation/main.py $i & done
wait
