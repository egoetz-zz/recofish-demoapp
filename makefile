.PHONY: environment
environment:
	pyenv install -s 3.11.4
	pyenv uninstall --force recofish-demoapp
	pyenv virtualenv 3.10.0 --force recofish-demoapp
	pyenv local recofish-demoapp

.PHONY: install
install:
	pip freeze | xargs -r pip uninstall -y && \
	pip install -r requirements.txt

.PHONY: run
run:
	export FLASK_ENV=development && flask run
