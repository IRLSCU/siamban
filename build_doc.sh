#!/bin/bash
cd ./docs

echo "start build doc in docs dir"

sphinx-apidoc -o source/modules/siamban ../siamban/
sphinx-apidoc -o source/modules/toolkit ../toolkit/
sphinx-apidoc -o source/modules/training_dataset ../training_dataset/
sphinx-apidoc -o source/modules/vot_siamban ../vot_siamban/
sphinx-apidoc -o source/modules/tools ../tools/

make clean
make html
