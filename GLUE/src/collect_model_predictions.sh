#!/bin/bash

LOG_FOLDER_NAME=$1
TO_FOLDER_NAME=$2

declare -A TASK_NAME_DICT=( ["ax"]="AX" \
				["cola"]="CoLA" \
				["mnli"]="MNLI-m" \
				["mnli-mm"]="MNLI-mm" \
				["mrpc"]="MRPC" \
				["qnli"]="QNLI" \
				["qqp"]="QQP" \
				["rte"]="RTE" \
				["sst2"]="SST-2" \
				["stsb"]="STS-B" \
				["wnli"]="WNLI")

echo "Copying prediction results from ${LOG_FOLDER_NAME} to ${TO_FOLDER_NAME}"

foldernames=`ls ${LOG_FOLDER_NAME}`
for folder in $foldernames
do
	if [ -d "${LOG_FOLDER_NAME}/${folder}" ]
	then
  		predict_files=`ls ${LOG_FOLDER_NAME}/${folder}/predict_results*.txt`
		for predict_file in $predict_files
		do
			task=$(basename $predict_file | awk -F'[_.]' '{print $3}')
			cp "${predict_file}" "${TO_FOLDER_NAME}/${TASK_NAME_DICT[${task}]}.tsv"
			echo "${TASK_NAME_DICT[${task}]} copied!"
		done
	fi
done  
