#!/bin/bash

python pt_evaluate_as_classifier.py -ds c15p -set test -gp
python pt_evaluate_as_classifier.py -ds ptbxl -set whole -gp
python pt_evaluate_as_classifier.py -ds ikem -set test -gp
python pt_evaluate_as_classifier.py -ds ptb -set test -gp
