.PHONY: text image multimodal check

text:
	python -m src.feature_extraction.text_extractor

image:
	python -m src.feature_extraction.image_extractor

multimodal:
	python -m src.feature_extraction.feature_fusion

check:
	tmux attach -t cluster

users:
	nvidia-smi --query-compute-apps=pid --format=csv,noheader | \
	xargs -I{} ps -o user= -p {} | sort | uniq