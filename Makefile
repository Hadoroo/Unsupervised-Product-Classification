.PHONY: text image multimodal check exp

text:
	python -m src.feature_extraction.text_extractor

image:
	python -m src.feature_extraction.image_extractor

multimodal:
	python -m src.feature_extraction.feature_fusion

exp:
	python -m src.model.experiments

check:
	tmux attach -t cluster

users:
	nvidia-smi --query-compute-apps=pid --format=csv,noheader | \
	xargs -I{} ps -o user= -p {} | sort | uniq