.PHONY: help start install uninstall reinstall
.DEFAULT: help

help:
	@echo "make start"
	@echo "          Run python project"
	@echo "----------"
	@echo "make install"
	@echo "          Install face analysis packages"
	@echo "----------"
	@echo "make uninstall"
	@echo "          Uninstall face analysis packages"
	@echo "----------"
	@echo "make reinstall"
	@echo "          Reinstall face analysis packages"
	@echo "----------"

start:
	poetry run start

install: 
	pip install poetry && poetry install

uninstall: 
	poetry cache clear pypi --all && rm -f poetry.lock

reinstall: uninstall install