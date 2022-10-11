#!/bin/bash

echo "Begin data binarization for OSRT optimality experiments"

echo "Threshold used for continuous feature: 4"

# python3 preprocessing.py ./airfoil/airfoil_self_noise.dat --sep \t -h  -A -b 4
#
# python3 preprocessing.py ./garment/garments_worker_productivity.csv -i -A -b 4
#
# python3 preprocessing.py ./optical/optical.csv --sep ';' --dec ',' -A -b 4
#
# python3 preprocessing.py ./real_estate/real_estate.csv -i -A -b 4
#
# python3 preprocessing.py ./seoul_bike/seoul_bike.csv -A -b 4
#
# python3 preprocessing.py ./servo/servo.data -A -b 4
#
# python3 preprocessing.py ./sync/synchronous_machine.csv --sep ';' --dec ',' -A -b 4
#
# python3 preprocessing.py ./yacht/yacht_hydrodynamics.data --sep ' ' -h -A -b 4

echo "Completed"
