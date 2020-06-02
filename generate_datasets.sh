python create_scenarios.py

for i in {0..1}
do
   python main.py $i & done
wait
