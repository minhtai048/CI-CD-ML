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

dev-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update