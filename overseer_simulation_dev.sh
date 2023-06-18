#!/bin/bash

for a in {a1,a2,a3,a4}; do
  python pt_evaluate_as_classifier.py -ds c15p -set dev -os -ip 10000 -es 1000 -dt 0.67 -lt 0.13 -a $a -r 10
  python pt_evaluate_as_classifier.py -ds ptbxl -set dev -os -ip 10000 -es 1000 -dt 0.67 -lt 0.13 -a $a -r 10
  python pt_evaluate_as_classifier.py -ds ikem -set dev -os -ip 10000 -es 1000 -dt 0.67 -lt 0.13 -a $a -r 10
  python pt_evaluate_as_classifier.py -ds ptb -set dev -os -ip 10000 -es 50 -dt 0.67 -lt 0.13 -a $a -r 10
done
