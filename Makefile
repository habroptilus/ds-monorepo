lint: 
	poetry run flake8 ./lilac
	poetry run mypy ./lilac 

format:
	poetry run isort ./lilac
	poetry run black ./lilac