#!/bin/bash

for x in `find . -type d`;
do
	mv ${x}*.txt ${x}
done
