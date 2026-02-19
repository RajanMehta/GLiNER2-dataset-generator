VENV   := .venv
PYTHON := $(VENV)/bin/python3
PID_FILE := .streamlit.pid

.PHONY: env up down

env:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip -q
	$(PYTHON) -m pip install -r requirements.txt

up:
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Streamlit is already running (PID $$(cat $(PID_FILE)))"; \
	else \
		$(VENV)/bin/streamlit run streamlit_app.py & echo $$! > $(PID_FILE); \
		echo "Streamlit started (PID $$(cat $(PID_FILE)))"; \
	fi

down:
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		kill $$(cat $(PID_FILE)) && rm -f $(PID_FILE); \
		echo "Streamlit stopped"; \
	else \
		rm -f $(PID_FILE); \
		echo "Streamlit is not running"; \
	fi
