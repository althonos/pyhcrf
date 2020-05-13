#!/bin/sh

set -e

. $(dirname $(dirname $0))/functions.sh

# --- Using proper Python executable -----------------------------------------

log Activating pyenv
eval "$(pyenv init -)"
pyenv shell $(pyenv versions --bare)

# --- Coverage ---------------------------------------------------------------

$PYTHON -m coverage combine
$PYTHON -m coverage xml
$PYTHON -m coverage report
