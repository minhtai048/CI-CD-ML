install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

format:
	black *.py

train:
	python train.py
	
eval:
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Model performance report](./history/current_result.png)' >> report.md
   
	cml comment create report.md

main-branch:
	git commit -am "new changes"
	git push origin main