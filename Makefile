lint: 
	poetry run flake8 ./lilac ./projects

format:
	poetry run isort ./lilac ./projects
	poetry run black ./lilac ./projects