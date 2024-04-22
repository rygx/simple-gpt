pip-compile -v --generate-hashes --output-file=requirements.txt requirements.in --allow-unsafe && \
pip-sync requirements.txt
