# VLP:
# A Visual-Linguistic Pretraining dataset
#  - COCO captions
#  - SBU Captions
#  - Visual Genome QA
#  - Conceptual captions
VLP: VLP/clip_features.hdf5 VLP/train.jsonl VLP/val.jsonl
VLP/clip_features.hdf5: COCO/train2017/000000000025.jpg COCO/val2017/000000227399.jpg SBUCaptions/sbu_images/29638548_ec868258f3.jpg VisualGenome/images/107899.jpg ConceptualCaptions/training/1 ConceptualCaptions/validation/1
	mkdir -p VLP

	# COCO
	cat COCO/annotations/captions_train2017.json | jq -c '.images[] | {"image_id": ("coco_" + (.id | tostring)), path: ("COCO/train2017/" + .file_name)}' > VLP/.data.jsonl
	cat COCO/annotations/captions_val2017.json | jq -c '.images[] | {"image_id": ("coco_" + (.id | tostring)), path: ("COCO/val2017/" + .file_name)}' >> VLP/.data.jsonl

	# SBUCaptions
	find SBUCaptions/sbu_images/ -name *.jpg | jq --raw-input -c '{image_id: ("sbu_" + match("/([a-z0-9_]+)\\.jpg").captures[0].string), path: .}' >> VLP/.data.jsonl

	# Visual Genome QA
	find VisualGenome/images -name *.jpg | jq --raw-input -c '{image_id: ("vgqa_" + match("/([0-9]+)\\.jpg").captures[0].string), path: .}' >> VLP/.data.jsonl

	# Conceptual Captions
	find ConceptualCaptions/training/ -type f | jq --raw-input -c '{image_id: ("cc_train_" + match("/([0-9]+)").captures[0].string), path: .}' >> VLP/.data.jsonl
	find ConceptualCaptions/validation/ -type f | jq --raw-input -c '{image_id: ("cc_val_" + match("/([0-9]+)").captures[0].string), path: .}' >> VLP/.data.jsonl

	# Extract features
	cat VLP/.data.jsonl | python ../../src/clip_bert/precompute_clip_visual_features.py --batch-size 1024 --output VLP/clip_features.hdf5
	rm VLP/.data.jsonl

VLP/train.jsonl VLP/val.jsonl: COCO/annotations/captions_train2017.json COCO/annotations/captions_val2017.json SBUCaptions/sbu-captions-all.json VisualGenome/question_answers.json ConceptualCaptions/Train_GCC-training.tsv ConceptualCaptions/Validation_GCC-1.1.0-Validation.tsv
	mkdir -p VLP
	
	# COCO
	cat COCO/annotations/captions_train2017.json | \
		jq -c '.annotations[] | {text: .caption, image_id: ("coco_" + (.image_id | tostring))}' > VLP/train.jsonl
	cat COCO/annotations/captions_val2017.json | \
		jq -c '.annotations[] | {text: .caption, image_id: ("coco_" + (.image_id | tostring))}' > VLP/val.jsonl

	# SBUCaptions (train + val)
	cat SBUCaptions/sbu-captions-all.json | jq -c '[.image_urls, .captions] | transpose | map( {image_url: .[0], caption:.[1] })[]' | \
		jq -c '{path: ("SBUCaptions/sbu_images/" + (.image_url | match("\\w+.jpg").string)), caption: .caption}' | python filter_nonexistent.py | \
		jq -c '{text: .caption, image_id: ("sbu_" + (.path | match("/([a-z0-9_]+)\\.jpg").captures[0].string))}' > VLP/sbu_captions.jsonl
	head -n 10000 VLP/sbu_captions.jsonl >> VLP/val.jsonl
	tail -n +10001 VLP/sbu_captions.jsonl >> VLP/train.jsonl
	rm VLP/sbu_captions.jsonl
	
	# Visual Genome QA
	cat VisualGenome/question_answers.json | jq -c '.[].qas[] | {text: (.question + " " +.answer), image_id: ("vgqa_" + (.image_id | tostring))}' | shuf > VLP/vgqa.jsonl
	head -n 10000 VLP/vgqa.jsonl >> VLP/val.jsonl
	tail -n +10001 VLP/vgqa.jsonl >> VLP/train.jsonl
	rm VLP/vgqa.jsonl

	# Conceptual Captions
	cat ConceptualCaptions/Train_GCC-training.tsv | cut -d'	' -f1 | jq --raw-input -sc 'split("\n") | to_entries[] | {text: .value, path: ("ConceptualCaptions/training/" + (.key+1 | tostring))}' | \
		python filter_nonexistent.py | jq -c '{text: .text, image_id: ("cc_train_" + (.path | match("/([0-9]+)").captures[0].string))}' >> VLP/train.jsonl
	cat ConceptualCaptions/Validation_GCC-1.1.0-Validation.tsv | cut -d'	' -f1 | jq --raw-input -sc 'split("\n") | to_entries[] | {text: .value, path: ("ConceptualCaptions/validation/" + (.key+1 | tostring))}' | \
                python filter_nonexistent.py | jq -c '{text: .text, image_id: ("cc_val_" + (.path | match("/([0-9]+)").captures[0].string))}' >> VLP/val.jsonl
