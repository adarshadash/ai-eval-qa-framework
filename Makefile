.PHONY: install test unit evals evals-live demo-bugs baseline clean

install:
	pip install -r requirements.txt

test:            ## all pytest gates (mock mode)
	pytest

unit:            ## metric + dataset unit tests only
	pytest -m unit

evals:           ## full runner: scores, reports, baseline regression check
	python -m evals.runner

evals-live:      ## real Claude API calls (needs ANTHROPIC_API_KEY)
	EVAL_MODE=live python -m evals.runner

demo-bugs:       ## inject bugs into the bot and watch the evals fail
	EVAL_INJECT_BUGS=1 python -m evals.runner; true

baseline:        ## accept current scores as the new baseline
	python -m evals.runner --update-baseline

clean:
	rm -rf reports/report.* .pytest_cache __pycache__ */__pycache__ */*/__pycache__
