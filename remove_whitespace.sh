#!/bin/bash

for f in figures/cone*mlab.png
do
	convert $f -trim $f
done

for f in figures/robustness*mlab.png
do
	convert $f -trim $f
done
